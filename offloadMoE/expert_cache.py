from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List
from collections import deque, defaultdict, OrderedDict
from .expert_wrapper import MixtralExpertWrapper

import torch
from torch import nn

ExpertUID = Any


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    index: int


@dataclass
class EvictionGroupInfo:
    # infos in main and offload devices; ordered from least recently used to most
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        for uid, info in self.main_infos.items():
            return info  # least recently used
        raise ValueError("No evictable experts")

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class ExpertCache:
    def __init__(self, make_module: callable, num_layer: int, main_size: int, offload_size: int, buffer_size: int):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        self.main_modules = [self._check_module(make_module()) for i in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = [
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(offload_size)]
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        # temporary storage to shave off latency
        self.device_expert_buffers = [deque([self._check_module(make_module()) for _ in range(buffer_size)]), 
                                    deque([self._check_module(make_module()) for _ in range(buffer_size)])]
        self.offloaded_storage_buffers = [deque([torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(buffer_size)]),
                                          deque([torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(buffer_size)])]
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)
        
        self.buffer_id = 0
        self.num_layer = num_layer

        # Store the active experts
        self.active_experts = [deque(), deque()]
        self.active_experts_idx = [deque(), deque()]

        self.H2D_stream = torch.cuda.Stream()
        self.D2H_stream = torch.cuda.Stream()

    def _check_module(self, module: MixtralExpertWrapper):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def add_expert(self, uid: ExpertUID, module: MixtralExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=False, index=i)
                    self.registered_experts[uid] = self.main_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found spot on device
        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, index=i)
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def set_pattern(self, pattern):
        # Batch, Seq, 32, 8
        self.pattern = pattern

    # Load experts in the first layer
    def load_first_layer(self):
        # Get expert info in sorted order [non-offloaded, offloaded]
        active_expert_id = self.pattern[0].nonzero().flatten().tolist()
        expert_id = [(0, expert_idx) for expert_idx in active_expert_id]
        expert_id = sorted(expert_id, key=lambda uid: self.registered_experts[uid].offloaded)
        expert_info = [self.registered_experts[id] for id in expert_id]
        
        # Update eviction group info
        eviction_group = self.group_infos[expert_info[0].eviction_group]
        for info in expert_info:
            eviction_group.mark_used(info)
        
        copy_idx = 0
        for i in range(len(expert_info)):
            if expert_info[i].offloaded == True:
                copy_idx = i
                break
            else:
                self.active_experts[self.buffer_id].append(self.main_modules[expert_info[i].index])
                self.active_experts_idx[self.buffer_id].append(expert_info[i].uid)

        while copy_idx < len(expert_info):
            self.active_experts[self.buffer_id].append(self.swap_double_buffer(expert_info[copy_idx], eviction_group.choose_expert_to_evict(), self.buffer_id))
            self.active_experts_idx[self.buffer_id].append(expert_info[copy_idx].uid)
            copy_idx += 1

    # Currently, we assume that all layers are accurate predictible
    def load_next_layer_expert(self, *uids, layer_id):
        self.D2H_stream.synchronize()
        self.H2D_stream.synchronize()

        # Last layer does not prefetch
        if layer_id == self.num_layer:
            return (self.active_experts[self.buffer_id], self.active_experts_idx[self.buffer_id])
        
        prev_buffer_id = self.buffer_id
        self.buffer_id ^= 1
        active_expert_id = self.pattern[layer_id].nonzero().flatten().tolist()
        expert_id = [(layer_id, expert_idx) for expert_idx in active_expert_id]
        expert_id = sorted(expert_id, key=lambda uid: self.registered_experts[uid].offloaded)
        expert_info = [self.registered_experts[id] for id in expert_id]

        # Update eviction group info
        eviction_group = self.group_infos[expert_info[0].eviction_group]
        for info in expert_info:
            eviction_group.mark_used(info)
        
        copy_idx = 0
        for i in range(len(expert_info)):
            if expert_info[i].offloaded == True:
                copy_idx = i
                break
            else:
                self.active_experts[self.buffer_id].append(self.main_modules[expert_info[i].index])
                self.active_experts_idx[self.buffer_id].append(expert_info[i].uid)

        while copy_idx < len(expert_info):
            self.active_experts[self.buffer_id].append(self.swap_double_buffer(expert_info[copy_idx], eviction_group.choose_expert_to_evict(), self.buffer_id))
            self.active_experts_idx[self.buffer_id].append(expert_info[copy_idx].uid)
            copy_idx += 1
        
        return (self.active_experts[prev_buffer_id], self.active_experts_idx[prev_buffer_id])

    def load_experts_overlap(self, *uids, unordered = False):
        if unordered:
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)
        
        # Case1: All experts in device are active
        # Case2: Part of experts in device are active
        # Case3: None of experts in device is active
        active_experts = deque()
        active_experts_idx = deque()
        copy_idx = 0
        for i in range(len(infos)):
            if infos[i].offloaded == True:
                copy_idx = i
                break
            else:
                active_experts.append(self.main_modules[infos[i].index])
                active_experts_idx.append(infos[i].uid)

        # If there is no active expert
        if len(active_experts) == 0:
            assert copy_idx == 0, "We should copy the expert from the beginning"
            expert_in_copy = self._swap(infos[copy_idx], eviction_group.choose_expert_to_evict())
            active_experts.append(expert_in_copy)
            active_experts_idx.append(infos[copy_idx].uid)
            copy_idx += 1
        
        while len(active_experts) != 0:
            # Wait the previous copy finish
            # torch.cuda.current_stream().wait_stream(self.D2H_stream)
            # torch.cuda.current_stream().wait_stream(self.H2D_stream)
            self.D2H_stream.synchronize()
            self.H2D_stream.synchronize()

            # Start the copy of next expert
            if copy_idx < len(infos) and copy_idx != 0:
                expert_in_copy = self._swap(infos[copy_idx], eviction_group.choose_expert_to_evict())
                active_experts.append(expert_in_copy)
                active_experts_idx.append(infos[copy_idx].uid)
                copy_idx += 1

            # Return the loaded expert
            yield(active_experts_idx.popleft(), active_experts.popleft())

    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        :example:
        >>> for uid, expert in expert_cache.load_experts(*list_of_uids, unordered=True):
        >>>     for uid, expert in expert_iter:
        >>>         result += expert(x) * get_moe_weight(uid)

        :param uids: iterate over the specified expert uids. Same uids as in add_expert
        :param unordered: if True, allows cache to iterate experts in arbitrary order
            The order is chosen to minimize the total wait time.
        :returns: an iterator that yields (uid, expert) pairs, only usable inside the for loop

        """
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:  # yield non-offloaded experts first
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)

        try:
            self.active = True
            # save pre-loaded experts before they can be swapped
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.index] for info in pre_loaded_infos])

            # begin loading experts into free buffers in background (via non-blocking copy)
            infos_to_load = deque([info for info in infos if info.offloaded])
            infos_in_loading = deque([])
            experts_in_loading = deque([])
            window_size = min(len(self.device_expert_buffers) - 1,
                              len(eviction_group.main_infos),
                              len(infos_to_load))
            for _ in range(window_size):
                info_to_load = infos_to_load.popleft()
                infos_in_loading.append(info_to_load)
                experts_in_loading.append(
                    self._swap(info_to_load, eviction_group.choose_expert_to_evict()))

            for info in infos:
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                elif len(infos_in_loading) > 0 and info is infos_in_loading[0]:
                    infos_in_loading.popleft()
                    yield (info.uid, experts_in_loading.popleft())
                    if len(infos_to_load) > 0:
                        info_to_load = infos_to_load.popleft()
                        infos_in_loading.append(info_to_load)
                        experts_in_loading.append(
                            self._swap(info_to_load, eviction_group.choose_expert_to_evict()))
                else:
                    raise RuntimeError("internal error: caching algorithm failed")
        finally:
            self.active = False

    def swap_double_buffer(self, info_to_load, info_to_evict, buffer_id):
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        offloaded_storage_buffer = self.offloaded_storage_buffers[buffer_id].popleft()
        device_expert_buffer = self.device_expert_buffers[buffer_id].popleft()
        
        with torch.cuda.stream(self.H2D_stream):
            device_expert_buffer.storage.copy_(self.offloaded_storages[info_to_load.index], non_blocking=True)

        with torch.cuda.stream(self.D2H_stream):
            offloaded_storage_buffer.copy_(self.main_modules[info_to_evict.index].storage, non_blocking=True)

        self.device_expert_buffers[buffer_id].append(self.main_modules[info_to_evict.index])
        self.main_modules[info_to_evict.index] = device_expert_buffer
        self.offloaded_storage_buffers[buffer_id].append(self.offloaded_storages[info_to_load.index])
        self.offloaded_storages[info_to_load.index] = offloaded_storage_buffer

        self.main_infos[info_to_evict.index] = info_to_load
        self.offloaded_infos[info_to_load.index] = info_to_evict
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_evict.index, info_to_load.index = info_to_load.index, info_to_evict.index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)

        return device_expert_buffer
    
    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        """Swap an offloaded expert (info_to_load) with an on-device expert (info_to_evict) return the loaded expert"""
        # print("To load ", info_to_load.offloaded)
        # print("To evict ", info_to_evict.offloaded)
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        offloaded_storage_buffer = self.offloaded_storage_buffers[0].popleft()
        device_expert_buffer = self.device_expert_buffers[0].popleft()
        
        with torch.cuda.stream(self.H2D_stream):
            device_expert_buffer.storage.copy_(self.offloaded_storages[info_to_load.index], non_blocking=True)

        with torch.cuda.stream(self.D2H_stream):
            offloaded_storage_buffer.copy_(self.main_modules[info_to_evict.index].storage, non_blocking=True)

        self.device_expert_buffers[0].append(self.main_modules[info_to_evict.index])
        self.main_modules[info_to_evict.index] = device_expert_buffer
        self.offloaded_storage_buffers[0].append(self.offloaded_storages[info_to_load.index])
        self.offloaded_storages[info_to_load.index] = offloaded_storage_buffer

        self.main_infos[info_to_evict.index] = info_to_load
        self.offloaded_infos[info_to_load.index] = info_to_evict
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_evict.index, info_to_load.index = info_to_load.index, info_to_evict.index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)

        return device_expert_buffer

    def prefetch(
            self,
            pattern_matrix,
    ) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        Pre-fetches experts based on a provided activation matrix for future layers.

        Args:
            pattern_matrix (torch.Tensor): A matrix indicating the activation state of experts across layers.

        Returns:
            Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]: Iterator for pre-fetched expert modules.
        """
        
        num_layers, num_experts = pattern_matrix.shape

        # 1. 统计出当前成 preloaded 和 offloaded 状态的 infos，根据 pattern_matrix 记录每个 expert 的状态：
        #  1) 无需操作：当 pattern_matrix[layer_id, expert_id] == 1 时,且 info.offloaded=False，该 expert 需要被用到，且已经 preload在 GPU 上
        #  1) 无需操作：当 pattern_matrix[layer_id, expert_id] == 0 时,且 info.offloaded=True，该 expert  不需要被用到，且已经 offload 在 GPU 上
        #  2) 需要 offload：当 pattern_matrix[layer_id, expert_id] == 0 时,且 info.offloaded=False，该 expert 不需要被用到，但已经 preload 到 GPU 上
        #  3）需要 preload: 当 pattern_matrix[layer_id, expert_id] == 1 时,且 info.offloaded=True，该 expert 需要被用到，但当前被 offload 到 CPU 上
        
        num_failed_preload = 0.
        for layer_id in range(num_layers):
            cpu2gpu_infos = []
            gpu2cpu_infos = []
            for expert_id in range(num_experts):
                uid: ExpertUID = (layer_id, expert_id)
                info = self.registered_experts.get(uid)
                
                # Skip if expert info is not found
                if info is None:
                    continue
                required_on_gpu = pattern_matrix[layer_id, expert_id] == 1
                if required_on_gpu and info.offloaded:
                    cpu2gpu_infos.append(info)
                elif not required_on_gpu and not info.offloaded:
                    gpu2cpu_infos.append(info)

            # Perform swaps
            while cpu2gpu_infos and gpu2cpu_infos:
                info_to_load = cpu2gpu_infos.pop()
                info_to_evict = gpu2cpu_infos.pop()
                # print(f"Swaping {info_to_load.uid}(cpu) to {info_to_evict.uid}(gpu)")
                self._swap(info_to_load, info_to_evict)
            
            # Todo: 支持在不同层之间的 expert 互相替换。
            # Todo: 因为每层的 expert 激活数量可能不一致，比如第一层只激活了 2 个，第二个需要激活 8 个（默认激活 4 个，offload4 个），那么已经 offload 的 4 个可以挪到第一层中去
            # Check remaining unprocessed experts due to imbalance in requirements
            # if len(cpu2gpu_infos) > 0:
            #     num_failed_preload += len(cpu2gpu_infos)
            #     failed_uids = [e.uid for e in cpu2gpu_infos]
            #     print(f"Layer{layer_id} has {len(cpu2gpu_infos)} experts {failed_uids} that cannot be preloaded.")
                    

import sys
from typing import List, Optional, Tuple, Union, Dict
import time
import os
import torch
import json
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers.modeling_outputs import MoEModelOutput

from offloadMoE.switch_transformer import build_offload_model


def prepare_data(dataset_list: Dict[str,int]):
    dataset_name = "tasksource/bigbench"
    names = list(dataset_list.keys())
    all_inputs = []
    for name in names:
        print(name)
        all_inputs.append(load_dataset(dataset_name, name))
    train_all_inputs = []
    # valid_all_inputs = []
    for dataset in all_inputs:
        train_all_inputs += [text for text in dataset["train"]["inputs"]]
        # valid_all_inputs += [text for text in dataset["validation"]["inputs"]]
    return train_all_inputs


def main(args):
    if os.environ.get('ipdb', False):
        from ipdb import set_trace
        set_trace()
    dataset_list = {
        "auto_categorization": 328,
        "tense": 286,
        "disfl_qa": 8000,
        "semantic_parsing_in_context_sparc": 1160,
        "word_sorting": 1900,
        "linguistics_puzzles": 2000,
    }
    print(f'Building dataset including {dataset_list}')
    data = prepare_data(dataset_list)
    ###### random order
    # indices = list(range(len(data)))
    # np.random.shuffle(indices)
    # data = np.array(data)[indices]
    ###### length-sorted order
    data = np.array(sorted(data, key=len))
    batch_size = 16
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    device = torch.device("cuda:0")
    model_name = "google/switch-base-16"
    state_path='~/.cache/huggingface/hub/models--google--switch-base-16/snapshots/*'
    state_path=os.path.expanduser(state_path)
    model = build_offload_model(
        offload_per_layer=8,
        buffer_size=8,
        state_path=state_path,
        model_name=model_name,
        device=device
    )
    model = model.to(device)
    max_new_tokens = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ###### baseline: original implementation
    if args.task == 0:
        run_benchmark(model, tokenizer, batches, max_new_tokens, device)

    else:
        raise NotImplementedError


def run_benchmark(model, tokenizer, batches, max_new_tokens, device):
    torch.cuda.reset_peak_memory_stats(device=device)
    num_tokens = []
    torch.cuda.synchronize()
    start = time.time()
    for batch_idx, batch in enumerate(batches):
        batch = batch.tolist()
        data = tokenizer(batch, return_tensors="pt", return_attention_mask=True, padding=True, truncation=True, max_length=512)
        data = {key: val.to(device) for key, val in data.items()}
        data['decoder_input_ids'] = torch.zeros(
            (data['input_ids'].shape[0],1), dtype=torch.long, device=device)
        batch_start = time.time()
        generated_token_ids, router_logits = custom_generate(
            **data, model=model, max_new_tokens=max_new_tokens
        )
        batch_end = time.time()
        num_tokens.append(generated_token_ids.numel())
    torch.cuda.synchronize()
    end = time.time()
    total_num_tokens = np.sum(num_tokens)
    throughput = total_num_tokens / (end - start)
    print(f"Throughput: {total_num_tokens} tokens/{end-start} sec = {throughput} tokens/s")
    peak_memory = torch.cuda.max_memory_reserved(device=device)
    print(f"Peak GPU Memory Usage: {peak_memory / 1024 ** 2:.2f} MB")
    time.sleep(10)


def custom_generate(
    input_ids,
    decoder_input_ids,
    attention_mask,
    model,
    max_new_tokens=128,
    past_key_values=None,
    temperature=0.9,
    top_p=0.9
):
    # 初始化生成的令牌列表和past_key_values（用于存储注意力层的状态，加速和优化生成）
    generated_tokens = []
    past = past_key_values
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        encoder_outputs = None
        for step in range(max_new_tokens):
            if step==0:
                # prefilling
                outputs = model(input_ids=input_ids,
                                decoder_input_ids=decoder_input_ids,
                                attention_mask=attention_mask,
                                past_key_values=past,
                                output_router_logits=True,
                                use_cache=True)  # use_cache允许模型返回past_key_values
            else:
                # decoding
                outputs = model(encoder_outputs=encoder_outputs,
                                decoder_input_ids=decoder_input_ids,
                                past_key_values=past,
                                output_router_logits=True,
                                use_cache=True)  # use_cache允许模型返回past_key_values
            # print(f"Step{step}: encoder-{outputs.encoder_router_logits[1][0].shape} decoder-{outputs.decoder_router_logits[1][0].shape}")
            # 获取输出中的下一个token logits和更新past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

            # 应用temperature来调整预测分布
            next_token_logits = next_token_logits / temperature
            filtered_logits = top_p_filtering(next_token_logits, top_p)
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)

            # 随机选择一个令牌
            next_token = torch.multinomial(probs, 1) # (batch_size , 1)
            # 将生成的令牌添加到列表和解码器输入中
            generated_tokens.append(next_token)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            encoder_outputs = MoEModelOutput(
                last_hidden_state=outputs.encoder_last_hidden_state,
                hidden_states=outputs.encoder_hidden_states,
                attentions=outputs.encoder_attentions,
                router_probs=outputs.encoder_router_logits,
            )

        return torch.cat(generated_tokens, dim=-1), (outputs.encoder_router_logits, outputs.decoder_router_logits)


def top_p_filtering(logits, top_p=0.9):
    """
    Filter a distribution of logits using nucleus (top-p) sampling

    Args:
    logits (torch.Tensor): The logits output by the model.
    top_p (float): The cumulative probability cutoff for nucleus sampling.

    Returns:
    torch.Tensor: The filtered logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def init_distributed_mode(args=SimpleNamespace()):
    def find_free_port(start_port: int, end_port: int):
        """
        Find a free port within the specified range.
        """
        for port in range(start_port, end_port):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", port))  # Try to bind to the port
                s.close()  # Close the socket if successful
                return port
            except OSError as e:
                # print(f"Port {port} is in use, trying next port.")
                continue
        raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and "LOCAL_RANK" in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = args.gpu
        args.dist_url = 'env://'
    else:
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = str(find_free_port(9000, 10000))
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        args.rank = 0
        args.gpu = args.local_rank = 0
        args.world_size = 1
        args.dist_url = 'env://'

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


if __name__ == '__main__':
    import argparse
    import torch.distributed as dist
    import fairscale.nn.model_parallel.initialize as fs_init
    
    def init_env():
        # define the model
        init_distributed_mode()
        fs_init.initialize_model_parallel(dist.get_world_size())

    init_env()
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Benchmark on a single GPU')

    # 添加参数
    parser.add_argument('--task', type=int, choices=[0, 1, 2, 3], default='0', help='Task to perform')
    # 0: running original implementation
    # 1: get and save pattern matrices for given bacthes of requests, including prefilling and decoding tokens
    # 2: run custom_generate with prefetched pattern matrices
    # 3: run custom_generate with pattern matrices predictor
    parser.add_argument('--pattern_matrices_path', type=str, default='pattern_matrices.pt', help='Path to pattern matrices')

    # 解析命令行输入
    args = parser.parse_args()
    main(args)

# torchrun --nproc_per_node=1 --master_port=26173  benchmark_single_gpu_switch.py --task 0
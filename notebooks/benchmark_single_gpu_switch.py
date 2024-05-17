import sys
from typing import List, Optional, Tuple, Union, Dict
import time
import os
import torch
import json
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
from transformers import TextStreamer

from hqq.core.quantize import BaseQuantizeConfig
from offloadMoE.build_model import OffloadConfig, QuantConfig, build_model
from offloadMoE.custom_layers import SparseMoeWrapper
from offloadMoE.switch_transformer import build_offload_model


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

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
    batch_size = 8
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    device = torch.device("cuda:0")
    model_name = "google/switch-base-16"
    state_path='/home/nus-hx/.cache/huggingface/hub/models--google--switch-base-16/snapshots/0ef7d88ed50ec5f2cfdc019e81cef04d19700f8f'
    model = build_offload_model(
        offload_per_layer=12,
        buffer_size= 6,
        state_path=state_path,
        model_name=model_name,
        device=device
    )
    model = model.to(device)
    max_new_tokens = 2
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ###### baseline: original implementation
    if args.task == 0:
        run_benchmark(model, tokenizer, batches, max_new_tokens, device)

    ###### get pattern matrices of given batches of requests, including prefilling and decoding tokens
    elif args.task ==1:
        pattern_matrices = get_pattern_matrices(model, tokenizer, batches, max_new_tokens, device)

    ###### Idea: run with ground truth of pattern matrices
    elif args.task == 2:
        os.environ['TRACE_PATTERN'] = "1"
        pattern_matrices = torch.load(args.pattern_matrices_path)
        run_benchmark_with_patterns(model, tokenizer, batch_size, max_new_tokens, device, pattern_matrices)

    elif args.task == 3:
        predictor = ...
        run_benchmark_with_predictor(model, tokenizer, batches, max_new_tokens, device, predictor)

    else:
        raise NotImplementedError

def run_benchmark(model, tokenizer, batches, max_new_tokens, device):
    num_tokens = []
    torch.cuda.synchronize()
    start = time.time()
    for batch_idx, batch in enumerate(batches):
        batch = batch.tolist()
        data = tokenizer(batch, return_tensors="pt", return_attention_mask=True, padding=True)
        data = {key: val.to(device) for key, val in data.items()}
        data['decoder_input_ids'] = torch.zeros(
            (data['input_ids'].shape[0],1), dtype=torch.long, device=device)
        num_tokens.append(data['input_ids'].numel())
        batch_start = time.time()
        generated_token_ids, router_logits = custom_generate(
            **data, model=model, max_new_tokens=max_new_tokens
        )
        batch_end = time.time()
    torch.cuda.synchronize()
    end = time.time()
    total_num_tokens = np.sum(num_tokens)
    throughput = total_num_tokens / (end - start)
    print(f"Throughput: {total_num_tokens} tokens/{end-start} sec = {throughput} tokens/s")


def get_pattern_matrices(model, tokenizer, batches, max_new_tokens, device):
    # Initialize a dictionary to hold the activations
    pattern_matrices = {
        # 0: {
        #     'prompt_text': "this is a prompt",
        #     'token_ids': [0, 2, 3, 56, 956, ...], # 大于等于 prompt
        #     'token_pattern_matrices': # 大小为(seq_len, num_layers, num_experts)的 one-hot 矩阵
        # },
        # 1: {},
        # ...
    }
    for batch_idx, batch in enumerate(batches):
        batch = batch.tolist()
        data = tokenizer(batch, return_tensors="pt", return_attention_mask=True, padding=True)
        data = {key: val.to(device) for key, val in data.items()}
        generated_token_ids, router_logits = custom_generate(
            data['input_ids'], data['attention_mask'], model, max_new_tokens=max_new_tokens, predictor=None
        )
        token_pattern_matrices_list = []
        for layer in model.modules():
            if isinstance(layer, SparseMoeWrapper):  # Check if the layer is a MoE layer
                token_pattern_matrices_list.append(layer.token_pattern_mask) # (bs, seq_len, num_experts)
        token_pattern_matrices_list = torch.stack(token_pattern_matrices_list, dim=-2) # (bs, seq_len, num_layers, num_experts)
        for i, text in enumerate(batch):
            pattern_matrices[len(batch)*batch_idx+i] = {
                'prompt_text': text,
                'prompt_token_ids': data['input_ids'][i].cpu(),
                'prompt_attention_mask': data['attention_mask'][i].cpu(),
                'token_ids': generated_token_ids[i].detach().cpu(),
                'token_pattern_matrices': token_pattern_matrices_list[i]
            }
        for layer in model.modules():
            if isinstance(layer, SparseMoeWrapper):  # Check if the layer is a MoE layer
                layer.token_pattern_mask = None
    torch.save(pattern_matrices, 'pattern_matrices.pt')
    return pattern_matrices


def run_benchmark_with_predictor(model, tokenizer, batches, max_new_tokens, device, predictor):
    # Initialize a dictionary to hold the activations
    num_tokens = []
    torch.cuda.synchronize()
    start = time.time()
    
    for batch_idx, batch in enumerate(batches):
        batch = batch.tolist()
        data = tokenizer(batch, return_tensors="pt", return_attention_mask=True, padding=True)
        data = {key: val.to(device) for key, val in data.items()}
        num_tokens.append(data['input_ids'].numel())
        batch_start = time.time()
        generated_token_ids, router_logits = custom_generate(
            data['input_ids'], data['attention_mask'], model, max_new_tokens=max_new_tokens, predictor=None
        )
        batch_end = time.time()
        print(f"Processing batch {batch_idx} data.input_ids.shape={data['input_ids'].shape} time costs: {batch_end-batch_start:.4f}s")
    torch.cuda.synchronize()
    end = time.time()
    total_num_tokens = np.sum(num_tokens)
    throughput = total_num_tokens / (end - start)
    print(f"Throughput: {total_num_tokens} tokens/{end-start} sec = {throughput} tokens/s")


def prefetch_experts_by_predictor(model, input_ids, attention_mask, predictor):
    if predictor == 'random': # for debug
        pattern_matrix = torch.randint(0, 2, (32, 8))
    else:
        ...
    
    for i, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.experts.prefetch(pattern_matrix)


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

    # 初始化生成的令牌列表和past_key_values（用于存储注意力层的状态，加速和优化生成）
    generated_tokens = []
    past = past_key_values
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for step in range(max_new_tokens):
            outputs = model(input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask,
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


def run_benchmark_with_patterns(model, tokenizer, batch_size, max_new_tokens, device, pattern_matrices):
    def prefetch_experts_by_pattern_matrices(model, pattern_matrix):
        for i, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.experts.prefetch(pattern_matrix)
            break

    get_batch_data = lambda key, batch: torch.stack([batch[i][key] for i in range(len(batch))], dim=0)

    def create_attention_mask(token_ids):
        # token_ids 是一个 (num_samples, seq_len) 的 PyTorch 张量
        seq_len = token_ids.size(1)
        
        # 找到每行第一个1出现的位置
        # cumsum 累积和将从第一个1开始生成非零值
        ones_and_zeros = (token_ids == 1).long()  # 将token等于1的位置变为1，其余为0
        cum_sum = torch.cumsum(ones_and_zeros, dim=1)
        
        # 生成 mask：cum_sum 大于0的位置表示这之后（包括该位置）应该是1
        attention_mask = cum_sum > 0

        return attention_mask.to(token_ids.device)

    def custom_generate_with_fixed_data(
        batch,
        model,
        max_new_tokens=128,
        past_key_values=None,
        temperature=0.9,
        top_p=0.9,
    ):
        model.eval()  # Put model in evaluation mode
        num_layers, num_experts = batch[0]['token_pattern_matrices'].shape[-2:]
        all_pattern_matrices = get_batch_data('token_pattern_matrices', batch) # (num_samples, prompt_len, 32, 8)
        all_token_ids = get_batch_data('token_ids', batch) # (num_samples, prompt_len+decoding_len)
        attention_mask = None
        with torch.no_grad():  # Disable gradient calculation
            # Initialize variables to store outputs and past_key_values
            generated_token_ids = None
            crt_tokens = None

            for token_index in range(max_new_tokens):
                if token_index == 0:
                    # prefilling
                    prompt_len = len(batch[0]['prompt_token_ids'])
                    pattern_matrices = all_pattern_matrices[:, :prompt_len, :, :] # (num_samples, prompt_len, 32, 8)
                    pattern_matrix = pattern_matrices.sum(0).sum(0) # (32, 8)
                    crt_tokens = all_token_ids[:, :prompt_len]
                    generated_token_ids = crt_tokens
                    attention_mask = create_attention_mask(crt_tokens)
                else:
                    # decoding
                    pattern_matrices = all_pattern_matrices[:, prompt_len+token_index-1, :, :] # (num_samples, 1, 32, 8)
                    pattern_matrix = pattern_matrices.sum(0) # (32, 8)
                    crt_tokens = all_token_ids[:, prompt_len+token_index-1].view(-1, 1)
                    attention_mask = torch.cat([attention_mask, torch.ones((len(batch), 1), device=attention_mask.device)], dim=-1)
                    generated_token_ids = torch.cat((generated_token_ids, crt_tokens), dim=1)
                prefetch_experts_by_pattern_matrices(
                    model, pattern_matrix
                )
                outputs = model(
                    input_ids=crt_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True  # Informs the model to return past key-values
                )

                # Update past_key_values for the next iteration
                past_key_values = outputs.past_key_values

                # Obtain logits
                logits = outputs.logits[:, -1, :] / temperature

                # Apply top-p nucleus sampling
                if top_p is not None:
                    filtered_logits = top_p_filtering(logits, top_p=top_p)
                else:
                    filtered_logits = logits
                probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)

                # Sample from the filtered distribution
                next_token_id = torch.multinomial(probabilities, num_samples=1)

                # Update the attention_mask for new token
                attention_mask2 = torch.cat([attention_mask, torch.ones((len(batch), 1), device=attention_mask.device)], dim=-1)

            return generated_token_ids

    # Initialize a dictionary to hold the activations
    # pattern_matrices = {
    #     0: {
    #         'prompt_text': "this is a prompt",
    #         'prompt_token_ids': [0, 2, 3, 56, 956, ...], # 大于等于 prompt
    #         'token_ids': [0, 2, 3, 56, 956, ...], # 大于等于 prompt
    #         'token_pattern_matrices': # 大小为(seq_len, num_layers, num_experts)的 one-hot 矩阵
    #     },
    #     1: {},
    #     ...
    # }
    num_tokens = []
    for i in range(len(pattern_matrices)):
        pattern_matrices[i]['token_ids'] = pattern_matrices[i]['token_ids'].to(device)
        pattern_matrices[i]['token_pattern_matrices'] = pattern_matrices[i]['token_pattern_matrices'].to(device)
    
    torch.cuda.synchronize()
    start = time.time()
    batch_indices = list(pattern_matrices.keys())
    batch_indices = [batch_indices[i:i + batch_size] for i in range(0, len(batch_indices), batch_size)]
    batches = [[pattern_matrices[i] for i in indices] for indices in batch_indices]

    for batch_idx, batch in enumerate(batches):
        batch_start = time.time()
        generated_token_ids = custom_generate_with_fixed_data(
            batch, model, max_new_tokens=max_new_tokens
        )
        batch_end = time.time()
        num_tokens.append(generated_token_ids.numel())
        print(f"Processing batch {batch_idx} generated_token_ids.shape={generated_token_ids.shape} time costs: {batch_end-batch_start:.4f}s")
    torch.cuda.synchronize()
    end = time.time()
    total_num_tokens = np.sum(num_tokens)
    throughput = total_num_tokens / (end - start)
    print(f"Throughput: {total_num_tokens} tokens/{end-start} sec = {throughput} tokens/s")


def test_custom_generate():
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    data = tokenizer(
        [
        'tell me a joke',
        'summary: this is a love story, where you and I live happily with the beautiful world',
        ], return_tensors='pt', return_attention_mask=True, padding=True
    )
    generated_token_ids, router_logits = custom_generate(
        data.input_ids.to('cuda'),
        data.attention_mask.to('cuda'),
        model.to('cuda'),
        max_new_tokens=128,
        temperature=0.9,
        top_p=0.9,
    )
    print(tokenizer.batch_decode(generated_token_ids.cpu().numpy().tolist(), skip_special_tokens=True))

    
if __name__ == '__main__':
    import argparse
    import torch.distributed as dist
    from accessory.util import misc
    import fairscale.nn.model_parallel.initialize as fs_init
    from glob import glob
    
    def init_env():
        # define the model
        misc.init_distributed_mode()
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
    # test_custom_generate()

# torchrun --nproc_per_node=1 --master_port=26173  benchmark_single_gpu_switch.py --task 0
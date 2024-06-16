import sys
from typing import List, Optional, Tuple, Union, Dict
import time
import os
import torch
import json
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
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
    if args.ipdb:
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
    data = data*3
    ###### random order
    # indices = list(range(len(data)))
    # np.random.shuffle(indices)
    # data = np.array(data)[indices]
    ###### length-sorted order
    data = np.array(sorted(data, key=len))
    batch_size = 8
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    device = torch.device("cuda:0")
    num_experts = args.num_experts
    model_name = f"google/switch-base-{num_experts}"
    state_path = f'~/.cache/huggingface/hub/models--google--switch-base-{num_experts}/snapshots/*'
    state_path = os.path.expanduser(state_path)
    model = build_offload_model(
        offload_per_layer=num_experts//4,
        buffer_size=num_experts//4,
        state_path=state_path,
        model_name=model_name,
        device=device
    )
    model = model.to(device)
    max_new_tokens = 32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

    ###### baseline: original implementation
    if args.task == 0:
        run_benchmark(model, tokenizer, batches, max_new_tokens, device)

    ###### get pattern matrices of given batches of requests, including prefilling and decoding tokens
    elif args.task ==1:
        pattern_matrices = get_pattern_matrices(model, tokenizer, batches, max_new_tokens, device, num_experts)

    ###### Idea: run with ground truth of pattern matrices
    elif args.task == 2:
        pattern_matrices = torch.load(args.pattern_matrices_path)
        run_benchmark_with_patterns(model, tokenizer, batch_size, max_new_tokens, device, pattern_matrices)

    elif args.task == 3:
        NUM_LABELS = 6 * num_experts
        pred_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        predictor = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
        predictor.lm_head = torch.nn.Linear(predictor.config.hidden_size, NUM_LABELS, bias=False)
        predictor.load_state_dict(torch.load(args.predictor_ckpt, map_location=torch.device('cpu')))
        predictor = predictor.to(device).bfloat16().eval()
        run_benchmark_with_predictor(model, tokenizer, batches, max_new_tokens, device, pred_tokenizer, predictor)

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
    top_p=0.9,
    predictor=None
):
    # 初始化生成的令牌列表和past_key_values（用于存储注意力层的状态，加速和优化生成）
    generated_tokens = [decoder_input_ids]
    past = past_key_values
    model.eval()  # Put model in evaluation mode
    with torch.inference_mode():  # Disable gradient calculation
        encoder_outputs = None
        encoder_router_indices = None
        decoder_router_indices_list = []
        for step in range(max_new_tokens):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=decoder_input_ids,
                            past_key_values=past,
                            output_router_logits=True,
                            use_cache=True)  # use_cache允许模型返回past_key_values
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
            decoder_input_ids = next_token
            encoder_outputs = MoEModelOutput(
                last_hidden_state=outputs.encoder_last_hidden_state,
                hidden_states=outputs.encoder_hidden_states,
                attentions=outputs.encoder_attentions,
                router_probs=outputs.encoder_router_logits,
            )
            if encoder_router_indices is None:
                encoder_router_logits = outputs.encoder_router_logits
                encoder_router_indices = [x[1] if len(x)==2 else None for x in encoder_router_logits]
            decoder_router_indices_list.append(outputs.decoder_router_logits)
        generated_tokens = torch.cat(generated_tokens, dim=-1) # (batch_size, seq_len)
        decoder_router_indices = []
        num_layers = len(decoder_router_indices_list[0])
        for i in range(num_layers):
            crt_layer_indices = None
            if i%2 ==1:
                crt_layer_indices = torch.cat([x[i][1] for x in decoder_router_indices_list], dim=1) # (batch_size, seq_len)
            decoder_router_indices.append(crt_layer_indices)
        return generated_tokens[:,:-1], (encoder_router_indices, decoder_router_indices)

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

def get_pattern_matrices(model, tokenizer, batches, max_new_tokens, device, num_experts):
    # Initialize a dictionary to hold the activations
    pattern_matrices = {
        # 0: {
        #     'prompt_text': "this is a prompt",
        #     'prompt_ids': [], # prompt token list
        #     'prompt_pattern': , # 大小为(seq_len, num_layers, num_experts)的 one-hot 矩阵
        #     'decode_ids': [], # deocde token list
        #     'decode_pattern': , # 大小为(seq_len, num_layers, num_experts)的 one-hot 矩阵
        # },
        # 1: {},
        # ...
    }
    for batch_idx, batch in enumerate(batches):
        if batch_idx % 100==0:
            print(f'Processing batch {batch_idx}/{len(batches)} for switch-{num_experts}')
        batch = batch.tolist()
        data = tokenizer(
            batch, return_tensors="pt", return_attention_mask=True, padding=True, truncation=True, max_length=256)
        data = {key: val.to(device) for key, val in data.items()}
        data['decoder_input_ids'] = torch.zeros(
            (data['input_ids'].shape[0], 1), dtype=torch.long, device=device)
        generated_token_ids, router_indices = custom_generate(
            **data, model=model, max_new_tokens=max_new_tokens
        )
        (encoder_router_indices, decoder_router_indices) = router_indices

        for i, text in enumerate(batch):
            prompt_ids = data['input_ids'][i].cpu()[data['attention_mask'][i].cpu()==1]
            decode_ids = generated_token_ids[i].detach().cpu()
            pad_len = (data['attention_mask'][i]==0).sum().item()
            pattern_matrices[len(batch)*batch_idx+i] = {
                'prompt_text': text,
                'prompt_ids': prompt_ids.tolist(),
                'decode_ids': decode_ids.tolist(),
                'prompt_pattern': [x[i, pad_len:].tolist() for x in encoder_router_indices if x is not None],
                'decode_pattern': [x[i].tolist() for x in decoder_router_indices if x is not None]
            }
    torch.save(pattern_matrices, 'switch_pattern_matrices.pt')
    hf_pattern_matrices = {
        'prompt_text': [],
        'prompt_ids': [],
        'decode_ids': [],
        'prompt_pattern': [],
        'decode_pattern': []
    }
    for i in range(len(pattern_matrices)):
        hf_pattern_matrices['prompt_text'].append(pattern_matrices[i]['prompt_text'])
        hf_pattern_matrices['prompt_ids'].append(pattern_matrices[i]['prompt_ids'])
        hf_pattern_matrices['decode_ids'].append(pattern_matrices[i]['decode_ids'])
        hf_pattern_matrices['prompt_pattern'].append(pattern_matrices[i]['prompt_pattern'])
        hf_pattern_matrices['decode_pattern'].append(pattern_matrices[i]['decode_pattern'])
    hf_pattern_matrices_dataset = Dataset.from_dict(hf_pattern_matrices)
    # hf_pattern_matrices_dataset.push_to_hub(f'marsggbo/bigbench4switch{num_experts}_pattern_truncation256')
    return pattern_matrices


def run_benchmark_with_patterns(model, tokenizer, batch_size, max_new_tokens, device, pattern_matrices):
    def prefetch_experts_by_pattern_matrices(model, pattern_matrix):
        for i, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.experts.prefetch(pattern_matrix)
            break

    def custom_generate_with_fixed_data(
        batch,
        model,
        max_new_tokens=128,
        past_key_values=None,
        temperature=0.9,
        top_p=0.9,
    ):
        model.eval()  # Put model in evaluation mode
        get_batch_data = lambda key, batch: torch.stack([batch[i][key] for i in range(len(batch))], dim=0)
        num_layers, num_experts = batch[0]['token_pattern_matrices'].shape[-2:]
        all_pattern_matrices = get_batch_data('token_pattern_matrices', batch) # (num_samples, prompt_len, 32, 8)
        all_token_ids = get_batch_data('token_ids', batch) # (num_samples, prompt_len+decoding_len)
        attention_mask = None
        with torch.no_grad():  # Disable gradient calculation
            # Initialize variables to store outputs and past_key_values
            generated_token_ids = None
            crt_tokens = None
            encoder_outputs = None

            for token_index in range(max_new_tokens):
                if token_index == 0:
                    # prefilling
                    prompt_len = len(batch[0]['prompt_token_ids'])
                    pattern_matrices = all_pattern_matrices[:, :prompt_len, :, :] # (num_samples, prompt_len, 32, 8)
                    pattern_matrix = pattern_matrices.sum(0).sum(0) # (32, 8)
                    crt_tokens = all_token_ids[:, :prompt_len]
                    generated_token_ids = crt_tokens
                    decoder_input_ids = torch.zeros((len(crt_tokens), 1), device=crt_tokens.device)
                    attention_mask = get_batch_data('prompt_attention_mask', batch).to(crt_tokens.device)
                    outputs = model(
                        input_ids=crt_tokens,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        past_key_values=past_key_values,
                        output_router_logits=True,
                        use_cache=True  # Informs the model to return past key-values
                    )
                else:
                    # decoding
                    pattern_matrices = all_pattern_matrices[:, prompt_len+token_index-1, :, :] # (num_samples, 32, 8)
                    pattern_matrix = pattern_matrices.sum(0) # (32, 8)
                    crt_tokens = all_token_ids[:, prompt_len+token_index-1].view(-1, 1)
                    attention_mask = torch.cat([attention_mask, torch.ones((len(batch), 1), device=attention_mask.device)], dim=-1)
                    generated_token_ids = torch.cat((generated_token_ids, crt_tokens), dim=1)
                    outputs = model(encoder_outputs=encoder_outputs,
                                    decoder_input_ids=decoder_input_ids,
                                    past_key_values=past_key_values,
                                    output_router_logits=True,
                                    use_cache=True)  # use_cache允许模型返回past_key_values
                # prefetch_experts_by_pattern_matrices(
                #     model, pattern_matrix
                # )

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
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
                encoder_outputs = MoEModelOutput(
                    last_hidden_state=outputs.encoder_last_hidden_state,
                    hidden_states=outputs.encoder_hidden_states,
                    attentions=outputs.encoder_attentions,
                    router_probs=outputs.encoder_router_logits,
                )
            return generated_token_ids

    # Initialize a dictionary to hold the activations
    # pattern_matrices = {
    #     0: {
    #         'prompt_text': "this is a prompt",
    #         'prompt_token_ids': [0, 2, 3, 56, 956, ...], # 大于等于 prompt
    #         'token_ids': [0, 2, 3, 56, 956, ...], # pad + prompt + decode
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


def run_benchmark_with_predictor(model, tokenizer, batches, max_new_tokens, device, pred_tokenizer, predictor):
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
    parser.add_argument('--predictor_ckpt', type=str, help='Path to predictor checkpoint')
    parser.add_argument('--num_experts', type=int, help='number of experts')
    parser.add_argument('--ipdb', action='store_true', help='Enable ipdb on error')

    # 解析命令行输入
    args = parser.parse_args()
    main(args)

# torchrun --nproc_per_node=1 --master_port=26173  benchmark_single_gpu_switch.py --task 0
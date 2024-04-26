import sys
from typing import List, Optional, Tuple, Union, Dict
import time
import torch
import json
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
from transformers import TextStreamer

from offloadMoE.build_model import OffloadConfig, build_model
from offloadMoE.custom_layers import SparseMoeWrapper
from offloadMoE.modeling_mixtral import build_offload_model

import os
import torch.distributed as dist
from accessory.util import misc
from glob import glob
import fairscale.nn.model_parallel.initialize as fs_init

def init_env():
    # define the model
    misc.init_distributed_mode()
    fs_init.initialize_model_parallel(dist.get_world_size())


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def prepare_data(dataset_list: Dict[str,int]):
    data = []
    # alpaca_data
    if 'alpaca' in dataset_list:
        alpaca_data = load_json("/home/scratch.shunkangz_gpu/Research/NUS_Project/Sequence-Scheduling/data/alpaca-train-10k.json")
        num_samples = dataset_list['alpaca']
        for i in range(num_samples):
            data.append(alpaca_data[i]['conversations'][0]['value'])

    # sst2
    if 'sst2' in dataset_list:
        sst2_data = load_dataset("stanfordnlp/sst2")['train'] # contain 67349 samples
        prefix_for_sst2 = '''For each given sentence, determine the sentiment expressed. If the sentiment is positive, return "positive". If the sentiment is negative, return "negative". Consider only these two categories for sentiment analysis. Please analyze the sentiment of the following sentence:'''
        num_samples = dataset_list['sst2']
        for i in range(num_samples):
            data.append(prefix_for_sst2 + sst2_data[i]['sentence'])

    # mrpc
    if 'mrpc' in dataset_list:
        mrpc_data  = load_dataset("SetFit/mrpc")["train"] # contain 3668 samples
        prefix_for_mrpc = '''Given two sentences, determine whether they express the same meaning. If they are paraphrases of each other, return "equivalent". If they are not, return "not equivalent". Please evaluate the following sentence pair:\n
        Sentence 1: "{}"
        Sentence 2: "{}"'''
        num_samples = dataset_list['mrpc']
        for i in range(num_samples):
            sample = mrpc_data[i]
            data.append(prefix_for_mrpc.format(sample['text1'], sample['text2']))

    # # yizhongw/self_instruct
    if 'yizhongw' in dataset_list:
        dataset = load_dataset("yizhongw/self_instruct", "super_natural_instructions")
        data_prompts = dataset['train']['prompt']
        num_samples = dataset_list['yizhongw']
        for i in range(num_samples):
            data.append(data_prompts[i])

    if 'tick666-math' in dataset_list:
        dataset = load_dataset("TICK666/Basic-Math-Chinese-1M-V1.1")['train'] # contains 1000000 samples
        num_samples = dataset_list['tick666-math']
        for i in range(num_samples):
            data.append(dataset[i]['text'])
    print(f"The data contains {len(data)} samples.")
    return data


def main():
    if os.environ.get('ipdb', False):
        from ipdb import set_trace
        set_trace()

    init_env()
    rank = dist.get_rank()

    dataset_list = {
        'alpaca': 1000,
        # 'sst2': 1000,
        # 'mrpc': 1000,
        # 'tick666-math': 1000,
        # 'yizhongw': 1000
    }
    print(f'Building dataset including {dataset_list}')
    batch_size = 1
    data = prepare_data(dataset_list)
    # indices = list(range(len(data)))
    # np.random.shuffle(indices)
    # data = np.array(data)[indices]
    data = np.array(sorted(data, key=len))

    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    config = AutoConfig.from_pretrained(
        model_name,
        num_local_experts=8,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    config.offload = True
    # config.num_hidden_layers = 1 # for debug only
    model = build_offload_model(config)
    model = model.bfloat16()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 8
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    num_tokens = []

    # Initialize a dictionary to hold the activations
    activations_record = {
        # 0: {
        #     'prompt_text': "this is a prompt",
        #     'token_ids': [0, 2, 3, 56, 956, ...], # 大于等于 prompt
        #     'token_pattern_matrices': # 大小为(seq_len, num_layers, num_experts)的 one-hot 矩阵
        # },
        # 1: {},
        # ...
    }
    torch.cuda.synchronize()
    start = time.time()
    for batch_idx, batch in enumerate(batches):
        batch = batch.tolist()
        data = tokenizer(batch, return_tensors="pt", return_attention_mask=True, padding=True)
        data = {key: val.to(device) for key, val in data.items()}
        num_tokens.append(data['input_ids'].numel())
        batch_start = time.time()
        generated_token_ids = custom_generate(
            data['input_ids'], data['attention_mask'], model, max_new_tokens=128, predictor=None
        )
        batch_end = time.time()
        token_pattern_matrices_list = []
        for layer in model.modules():
            if isinstance(layer, SparseMoeWrapper):  # Check if the layer is a MoE layer
                token_pattern_matrices_list.append(layer.token_pattern_mask) # (bs, seq_len, num_experts)
        token_pattern_matrices_list = torch.stack(token_pattern_matrices_list, dim=-1) # (bs, seq_len, num_layers, num_experts)
        for i, text in enumerate(batch):
            activations_record[len(batch)*batch_idx+i] = {
                'prompt_text': text,
                'prompt_token_ids': data['input_ids'][i].cpu(),
                'token_ids': generated_token_ids[i].detach().cpu(),
                'token_pattern_matrices': token_pattern_matrices_list[i]
            }
        for layer in model.modules():
            if isinstance(layer, SparseMoeWrapper):  # Check if the layer is a MoE layer
                layer.token_pattern_mask = None
        # print(tokenizer.batch_decode(result.cpu().numpy().tolist(), skip_special_tokens=True))
        print(f"Processing batch {batch_idx} data.input_ids.shape={data['input_ids'].shape} time costs: {batch_end-batch_start:.4f}s")
    torch.cuda.synchronize()
    end = time.time()
    torch.save(activations_record, 'activations_record.pt')
    total_num_tokens = np.sum(num_tokens)
    throughput = total_num_tokens / (end - start)
    print(f"Throughput: {total_num_tokens} tokens/{end-start} sec = {throughput} tokens/s")


def prefetch_experts_by_predictor(model, input_ids, attention_mask, predictor):
    if predictor == 'random':
        pattern_matrix = torch.randint(0, 2, (32, 8))
    else:
        ...
    
    for i, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.experts.prefetch(pattern_matrix)

def custom_generate(
    input_ids,
    attention_mask,
    model,
    max_new_tokens=128,
    past_key_values=None,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    predictor=None
):
    """
    Generate text from an input using caching and sampling techniques.

    Args:
    input_ids (torch.Tensor): Tensor of token ids to be fed to the model.
    attention_mask (torch.Tensor): Tensor representing the attention mask.
    model (transformers.PreTrainedModel): The model to use for generating text.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
    max_new_tokens (int): Maximum number of tokens to generate.
    temperature (float): Sampling temperature for controlling generation randomness.
    top_p (float): Nucleus sampling cutoff probability.

    Returns:
    torch.Tensor: Tensor containing the generated token ids.
    """        
    global activations_record
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        # Initialize variables to store outputs and past_key_values
        generated_token_ids = input_ids
        crt_tokens = input_ids

        for _ in range(max_new_tokens):
            if predictor is not None:
                prefetch_experts_by_predictor(
                    model, generated_token_ids, attention_mask, predictor
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
            if top_k is not None:
                filtered_logits = top_k_filtering(logits, top_k=top_k)
            elif top_p is not None:
                filtered_logits = top_p_filtering(logits, top_p=top_p)
            else:
                filtered_logits = logits
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)

            # Sample from the filtered distribution
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            crt_tokens = next_token_id
            generated_token_ids = torch.cat((generated_token_ids, next_token_id), dim=1)

            # Update the attention_mask for new token
            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=attention_mask.device)], dim=-1)

        return generated_token_ids

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

def top_k_filtering(logits, top_k=10):
    """
    Filter a distribution of logits using top-k sampling

    Args:
    logits (torch.Tensor): The logits output by the model.
    top_k (int): The number of top elements to keep.

    Returns:
    torch.Tensor: The filtered logits.
    """
    top_k_values, top_k_indices = torch.topk(logits, top_k, sorted=False)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, top_k_indices, False)
    logits[mask] = float('-inf')
    return logits


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
    generated_token_ids = custom_generate(
        data.input_ids.to('cuda'),
        data.attention_mask.to('cuda'),
        model.to('cuda'),
        max_new_tokens=128,
        temperature=0.9,
        top_p=0.9,
    )
    print(tokenizer.batch_decode(generated_token_ids.cpu().numpy().tolist(), skip_special_tokens=True))

    
if __name__ == '__main__':
    main()
    # test_custom_generate()
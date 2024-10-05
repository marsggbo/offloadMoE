import sys
from typing import List, Optional, Tuple, Union, Dict
import time
import os
import torch
import json
from glob import glob
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.modeling_outputs import MoEModelOutput
from datasets import load_dataset, Dataset


def load_batch(dataset, batch_indices, tokenizer):
    samples = dataset.select(batch_indices)
    prompts = samples["prompt_text"]
    prompt_data = tokenizer(
        prompts, padding=True, return_tensors="pt", return_attention_mask=True)
    prompt_ids, attention_mask = list(prompt_data.values())
    decode_ids = torch.tensor(samples["decode_ids"]).long()
    return prompt_ids, attention_mask, decode_ids


def main(args):
    if args.ipdb:
        from ipdb import set_trace
        set_trace()
    device = torch.device("cuda")
    data_name = args.data_name
    home_path = os.path.expanduser('~')
    if 'switch' in args.predictor_ckpt.lower():
        num_experts = args.num_experts
        NUM_LABELS = 6 * num_experts
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        dataset_name = f"marsggbo/{data_name}_switch{num_experts}_token_patterns"
        predictor_name = f'marsggbo/t5-small_dff2048_dmodel32_token-pattern-predictor_switch{num_experts}_{data_name}'
    else:
        num_experts = 8
        NUM_LABELS = 32 * num_experts
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        tokenizer.pad_token = tokenizer.eos_token
        dataset_name = f"marsggbo/{data_name}_mixtral8x7bInstructv0.1_token_patterns"
        predictor_name = f'marsggbo/t5-small_dff2048_dmodel32_token-pattern-predictor_mixtral8x7bInstructv0.1_{data_name}'
    tokenizer.padding_side='left'
    ckpt_path = f"{home_path}/.cache/huggingface/hub/*{predictor_name.split('/')[-1]}/snapshots/*/*bin"
    to_load_ckpt = True
    try:
        ckpt_path = glob(ckpt_path)[0]
    except:

        print(f"No checkpoint found for {predictor_name}")
        to_load_ckpt = False
        exit()

    # predictor_name = "marsggbo/t5-small_dff2048_dmodel32_token-pattern-predictor_switch64_wmt16" # for test
    predictor_config = AutoConfig.from_pretrained(predictor_name)
    predictor = AutoModelForSeq2SeqLM.from_config(config=predictor_config)
    predictor.lm_head = torch.nn.Linear(predictor.config.hidden_size, NUM_LABELS, bias=False)
    if to_load_ckpt:
        predictor.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
        print(f"Loaded predictor from {ckpt_path}")
    predictor = predictor.to(device).bfloat16().eval()

    new_dataset_name = dataset_name.replace('_token_patterns', '_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32')
    origin_dataset = load_dataset(dataset_name)['train']
    dataset = origin_dataset.shard(num_shards=len(origin_dataset)//10000, index=0)
    indices = list(range(len(dataset)))
    batch_size = 16
    batch_indices_list = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    decoder_patterns = []
    for j, batch_indices in enumerate(batch_indices_list):
        if j%100==0:
            print(f"{j}/{len(batch_indices_list)}")
        batch_data = load_batch(dataset, batch_indices, tokenizer)
        batch_data = [x.to(device) for x in batch_data]
        prompt_ids, attention_mask, decode_ids = batch_data
        past_key_values = None
        encoder_outputs = None
        num_steps = decode_ids.shape[1]
        batch_decode_patterns = []
        topk = 4
        with torch.inference_mode():
            for step in range(num_steps):
                outputs = predictor(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decode_ids[:, step].view(-1,1),
                    past_key_values=past_key_values,
                    encoder_outputs=encoder_outputs,
                )
                past_key_values = outputs.past_key_values
                if encoder_outputs is None:
                    encoder_outputs = MoEModelOutput(
                        last_hidden_state=outputs.encoder_last_hidden_state,
                        hidden_states=outputs.encoder_hidden_states,
                        attentions=outputs.encoder_attentions,
                    )
                logits = outputs.logits # (bs, 1, NUM_LABELS)
                logits = logits.view(len(batch_indices), 1, -1, num_experts) # (bs, 1, #layers, num_experts)
                top_indices = logits.topk(topk, dim=-1)[1] # (bs, 1, #layers, topk)
                batch_decode_patterns.append(top_indices.cpu())
            batch_decode_patterns = torch.cat(batch_decode_patterns, dim=1) # (bs, seq_len, #layers, topk)
            decoder_patterns.append(batch_decode_patterns)
    decoder_patterns = torch.cat(decoder_patterns, dim=0) # (num_samples, seq_len, #layers, topk)
    new_dataset = dataset.add_column('predictor_pattern', decoder_patterns.tolist())
    new_dataset.push_to_hub(new_dataset_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark on a single GPU')

    # 添加参数
    parser.add_argument('--data_name', type=str, default='xsum', help='dataset name: xsum or wmt16')
    parser.add_argument('--predictor_ckpt', type=str, help='Path to predictor checkpoint')
    parser.add_argument('--num_experts', type=int, default=32, help='number of experts per Switch MoE layer')
    parser.add_argument('--ipdb', action='store_true', help='Enable ipdb on error')

    # 解析命令行输入
    args = parser.parse_args()
    main(args)

# python -m ipdb gen_predictor_pattern.py --predictor_ckpt 
import os
import types
import logging
import pathlib
import typing
import random
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn as nn
from datasets import load_dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    Trainer,
    BitsAndBytesConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

num_decoder_sparse_layer = 6 # switch-32/64/128/256: 6; mixtral: 32
num_experts_per_layer = 32 # mixtral: 8
NUM_LABELS = num_decoder_sparse_layer * num_experts_per_layer
PADDING_SIDE = 'left'
model_name_or_path = "google-t5/t5-small"

def create_optimizer_and_scheduler(
    model, num_training_steps, learning_rate_head=2e-4, learning_rate_base=2e-5, warmup_steps=5000, weight_decay=0.01):
    # 分离 `lm_head` 的参数和其它参数
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "lm_head" not in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-4,
            "lr": learning_rate_base,
        },
        {
            "params": [p for n, p in model.named_parameters() if "lm_head" not in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": learning_rate_base,
        },
        {
            "params": [p for n, p in model.named_parameters() if "lm_head" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": learning_rate_head,
        },
        {
            "params": [p for n, p in model.named_parameters() if "lm_head" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": learning_rate_head,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=model_name_or_path)
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="left", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=f"marsggbo/xsum_switch{num_experts_per_layer}_token_patterns", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class LoraArguments:
    use_lora: bool = field(default=False)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: [
            # switch
            "q",
            "k",
            "v",
            "o",
            "wi",
            "wo"
            
            # # mixtral
            # "q_proj",
            # "v_proj",
            # "k_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    inference_mode: bool = False

@dataclass
class CustomArguments:
    eval_only: bool = field(default=False, metadata={"help": "Evaluation only"})
    ckpt_path: str = field(default=None, metadata={"help": "The checkpoint path for evaluation"})
    eval_max_seq_size: int = field(default=512, metadata={"help": "eval truncation size"})
    train_max_seq_size: int = field(default=512, metadata={"help": "train truncation size"})
    lr_head: float = field(default=2e-4, metadata={"help": "learning rate for head"})
    lr_base: float = field(default=2e-5, metadata={"help": "learning rate for base"})
    dim_ff: float = field(default=None, metadata={"help": "dimension ffn"})
    dim_model: float = field(default=None, metadata={"help": "dimension of model"})
    new_num_layers: float = field(default=None, metadata={"help": "number of layers"})
    ipdb: bool = field(default=False, metadata={"help": "debug with ipdb"})
    suffix: str = field(default=None, metadata={"help": "suffix of experiment name"})


# 定义 MoEPatternDataset 类
class MoEPatternDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        training=False,
        train_max_seq_size = 512,
        eval_max_seq_size = 512,
        num_experts_per_layer=num_experts_per_layer,
    ):
        self.data = dataset
        self.training = training
        self.truncate_ratio = 1.
        self.train_max_seq_size = train_max_seq_size
        self.eval_max_seq_size = eval_max_seq_size
        self.num_experts_per_layer = num_experts_per_layer

    def __len__(self):
        return len(self.data)
        # return 128
    
    def __getitem__(self, idx):
        seq_data = self.data[idx]
        input_ids = seq_data['prompt_ids'] # (encode_seq_len,)
        input_ids = torch.tensor(input_ids, dtype=int)
        attention_mask = torch.ones(len(input_ids)) # (encode_seq_len,)
        decoder_input_ids = torch.tensor(seq_data['decode_ids'], dtype=int)
        labels = np.stack(seq_data['decode_pattern']) # (#layers, decode_seq_len)
        if len(labels.shape)==3:
            labels = torch.from_numpy(labels).permute(1, 0, 2) # (decode_seq_len, #layers, topk_expert_indices)
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_experts_per_layer).float() # (decode_seq_len, #layers, topk, #experts)
            labels = labels.sum(-2)
        else:
            labels = torch.from_numpy(labels).permute(1, 0) # (decode_seq_len, #layers)
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_experts_per_layer).float() # (decode_seq_len, #layers, #experts)
        decode_seq_len = len(decoder_input_ids)

        if self.training:
            self.truncate_ratio = random.uniform(0.2, 1)
            if self.train_max_seq_size is not None:
                truncate_length = min(int(decode_seq_len * self.truncate_ratio), self.train_max_seq_size)
            else:
                truncate_length = int(decode_seq_len * self.truncate_ratio)
            start_index = random.randint(0, decode_seq_len - truncate_length)
        else:
            if self.eval_max_seq_size is not None:
                truncate_length = min(int(decode_seq_len * self.truncate_ratio), self.eval_max_seq_size)
            else:
                truncate_length = int(decode_seq_len * self.truncate_ratio)
            start_index = 0
        start_index = 0
        end_index = start_index + truncate_length
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids[start_index: end_index],
            'labels': labels[start_index:end_index]
        }


@dataclass
class PatternDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        if not hasattr(self, 'padding_side'):
            self.padding_side = PADDING_SIDE
        assert self.padding_side in ['left', 'right'], "Padding should be on one side (left or right)"
        non_label_features =[]
        for feature in features:
            item = {key: val for key, val in feature.items() if key in ['input_ids', 'attention_mask']}
            non_label_features.append(item)
        batch = super().__call__(non_label_features)
        bs = len(features)
        decoder_input_ids = [feature['decoder_input_ids'] for feature in features]
        label_features = [feature['labels'].view(-1, NUM_LABELS) for feature in features] # (decode_seq_len, #layers*#experts)
        batch['decoder_input_ids'] = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=0)
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(
            label_features, batch_first=True, padding_value=0).view(bs, -1, NUM_LABELS)
        return batch

def top_1_one_hot(logits):
    batch_size, seq_len, num_layer, num_experts = logits.shape
    
    # 获取top-1索引，values不使用
    _, top_1_indices = logits.topk(1, dim=-1, largest=True, sorted=True)  # shape: (batch_size, seq_len, num_layer, 2)
    
    # 初始化一个全0的tensor
    one_hot = torch.zeros_like(logits)  # shape: (batch_size, seq_len, num_layer, num_experts)
    
    # 生成one hot矩阵
    batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1).expand(-1, seq_len, num_layer, 2)
    seq_indices = torch.arange(seq_len).view(1, seq_len, 1, 1).expand(batch_size, -1, num_layer, 2)
    layer_indices = torch.arange(num_layer).view(1, 1, num_layer, 1).expand(batch_size, seq_len, -1, 2)
    expert_indices = top_1_indices  # 直接使用top-2索引
    
    # 利用上面生成的索引将对应位置设为1
    one_hot[batch_indices, seq_indices, layer_indices, expert_indices] = 1

    return one_hot

def get_acc(logits, labels, verbose=False):
    bs, seq_len, num_layer, num_experts = logits.shape
    y_true = labels
    y_pred = logits
    # y_true = labels.view(bs, seq_len, -1) # (bs, seq_len, #layers*#experts)
    # y_pred = logits.view(bs, seq_len, -1) # (bs, seq_len, #layers*#experts)
    
    y_true_prefill = y_true[:, :-63, ...] # (bs, num_prompts, #layers, #experts)
    y_true_prefill = y_true_prefill.sum(1).unsqueeze(1) # (bs, 1, #layers, #experts)
    y_true_decode = y_true[:, -63:, ...]
    y_true = torch.cat([y_true_prefill, y_true_decode], dim=1) # (bs, 1+num_decode, #layers, #experts)
    y_true = y_true.permute(1,0,2,3) # (1+num_decode, bs, #layers, #experts)
    y_true = y_true.sum(1) # (1+num_decode, #layers, #experts)

    y_pred_prefill = y_pred[:, :-63, ...]
    y_pred_prefill = y_pred_prefill.sum(1).unsqueeze(1) # (bs, 1, #layers, #experts)
    y_pred_decode = y_pred[:, -63:, ...]
    y_pred = torch.cat([y_pred_prefill, y_pred_decode], dim=1) # (bs, 1+num_decode, #layers, #experts)
    y_pred = y_pred.permute((1,0,2,3)) # (1+num_decode, bs, #layers, #experts)
    y_pred = y_pred.sum(1) # (1+num_decode, #layers, #experts)


    mask = (y_true > 0) & (y_pred > 0) # (1+num_decode, #layers, #experts)
    accuracy = mask.float().mean(-1).mean()
    if verbose:
        print(f"ACC={mask.sum()}/{mask.numel()}={accuracy} (labels.shape={labels.shape})")

    return accuracy

def new_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output) # (bs, seq_len, num_layers*num_experts)
        loss = None
        if labels is not None:
            # masked BCE loss
            labels = labels.to(lm_logits.device).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                lm_logits.view(-1, num_experts_per_layer),
                labels.view(-1, num_experts_per_layer),
                reduction='none')
            loss_mask = labels.view(-1, num_experts_per_layer).sum(-1) != 0
            loss = loss[loss_mask].sum() / loss_mask.sum()
            # if not self.training:
            #     preds = top_1_one_hot(lm_logits)
            #     print('predicted case')
            #     acc = get_acc(preds, labels, 1)
            #     random_logits = torch.rand_like(lm_logits)
            #     rand_preds = top_1_one_hot(random_logits)
            #     print('random case')
            #     acc = get_acc(rand_preds, labels, 1)
            #     # print(acc)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def acc_precision_recall_f1(y_true, y_pred):
    # 真正例 (True Positives)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    y_true = y_true.reshape(-1, NUM_LABELS)
    y_pred = y_pred.reshape(-1, NUM_LABELS)
    print(f"origin y_true.shape={y_true.shape}")
    indices = np.any(y_true, axis=-1)
    y_true = y_true[indices]
    y_pred = y_pred[indices]
    print(f"filtered y_true.shape={y_true.shape}")

    # 准确率
    num_tokens = y_true.shape[0]
    accuracy = TP / (num_tokens*num_decoder_sparse_layer)
    print(f"non-padding ratio: {indices.sum()}/{len(indices)}={indices.sum()/len(indices)}\n")

    return {
        'accuracy': accuracy,
    }

def compute_metrics(outputs):
    true_labels = outputs.label_ids
    pred_labels = outputs.predictions
    if isinstance(pred_labels, tuple):
        pred_labels = pred_labels[0]
    if len(pred_labels.shape) == 3:
        origin_bs, seq_len, dim = pred_labels.shape
    elif len(pred_labels.shape)==4:
        origin_bs, seq_len, num_layer, num_experts = pred_labels.shape
        dim = num_layer * num_experts
    true_labels = true_labels.reshape(-1, num_experts_per_layer)
    pred_labels = pred_labels.reshape(-1, num_experts_per_layer)
        
    # Convert predictions to top-2 one-hot encoding
    top1_preds_one_hot = np.zeros_like(pred_labels)
    top2_preds_one_hot = np.zeros_like(pred_labels)
    sort_indices = np.argsort(pred_labels, axis=1)
    top1_indices = sort_indices[:, -1:]
    top2_indices = sort_indices[:, -2:]
    rows = np.arange(pred_labels.shape[0])[:, None]
    top1_preds_one_hot[rows, top1_indices] = 1
    top2_preds_one_hot[rows, top2_indices] = 1
    top1_acc = acc_precision_recall_f1(
        true_labels, top1_preds_one_hot
    )['accuracy']
    top2_acc = acc_precision_recall_f1(
        true_labels, top2_preds_one_hot
    )['accuracy']
    token_metrics = {
        'top1@acc': top1_acc,
        'top2@acc': top2_acc
    }

    onehot_true_labels = true_labels.reshape(origin_bs, seq_len, num_decoder_sparse_layer, num_experts_per_layer)
    batch_metrics = {}
    for topk, onehot_pred_labels in enumerate([top1_preds_one_hot, top2_preds_one_hot]):
        batch_metrics[f'top{topk+1}'] = {}
        onehot_pred_labels = onehot_pred_labels.reshape(origin_bs, seq_len, num_decoder_sparse_layer, num_experts_per_layer)
        for bs in [4, 8, 16, 32]:
            if bs > len(true_labels):
                continue
            num_batches = len(onehot_true_labels) // bs
            batch_acc_list = []
            for batch_idx in range(num_batches):
                crt_batch_acc = []
                for token_idx in range(seq_len):
                    x_true = onehot_true_labels[batch_idx*bs:(batch_idx+1)*bs, token_idx].sum(0)
                    x_pred = onehot_pred_labels[batch_idx*bs:(batch_idx+1)*bs, token_idx].sum(0)
                    true_positives = np.sum(np.logical_and(x_true > 0, x_pred > 0), axis=1)
                    actual_positives = np.sum(x_true>0, axis=1)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        layer_accuracies = np.where(actual_positives > 0, true_positives / actual_positives, 1)
                    crt_batch_acc.append(layer_accuracies.mean())
                batch_acc_list.append(crt_batch_acc)
            batch_metrics[f'top{topk+1}'][f'bs{bs}_acc'] = np.mean(batch_acc_list)
        batch_metrics.update(token_metrics)
    print(batch_metrics)
    return batch_metrics
    

def train():
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, transformers.TrainingArguments, LoraArguments, CustomArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        custom_args
    ) = parser.parse_args_into_dataclasses()
    model_name = model_args.model_name_or_path.split('/')[-1]
    dataset_name = data_args.data_path.split('/')[-1]
    lr_head = custom_args.lr_head
    lr_base = custom_args.lr_base
    wd = training_args.weight_decay
    bs = training_args.per_device_train_batch_size
    run_name = f"{model_name}_{dataset_name}_lrhead{lr_head}_lrbase{lr_base}_ws{wd}_bs{bs}_seed{training_args.seed}"
    if custom_args.dim_ff:
        dim_ff = int(custom_args.dim_ff)
        dim_model = int(custom_args.dim_model)
        run_name += f'_dff{dim_ff}_dmodel{dim_model}'
    elif custom_args.new_num_layers:
        new_num_layers = int(custom_args.new_num_layers)
        run_name += f'_{new_num_layers}layers'
    if custom_args.suffix:
        run_name += f"_{custom_args.suffix}"
    training_args.run_name = run_name
    output_dir = training_args.output_dir
    if training_args.run_name:
        output_dir += f'{training_args.run_name}'
        training_args.output_dir = output_dir
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
    print(f'Model args: {model_args}')
    print(f'Data args: {data_args}')
    print(f'Lora args: {lora_args}')
    print(f'Custom args: {custom_args}')
    print(f'Training args: {training_args}')
    if custom_args.ipdb:
        from ipdb import set_trace
        set_trace()
    
    ################################
    # 实例化 tokenizer 和 model
    ################################
    quantization_config = None
    if lora_args.use_lora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    if custom_args.dim_ff:
        assert custom_args.dim_model is not None
        # ############### width ###############
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        model_config.d_ff = int(custom_args.dim_ff) # 2048 by default
        model_config.d_model = int(custom_args.dim_model) # 512 by default
        model = AutoModelForSeq2SeqLM.from_config(model_config)    
    elif custom_args.new_num_layers:
        ############### depth ###############
        model_name = model_args.model_name_or_path
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_layers = int(custom_args.new_num_layers)
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
        for i in range(3):
            model.encoder.block[i].load_state_dict(base_model.encoder.block[i].state_dict())
            model.decoder.block[i].load_state_dict(base_model.decoder.block[i].state_dict())
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )

    model.forward = types.MethodType(new_forward, model)
    model.lm_head = nn.Linear(model.config.hidden_size, NUM_LABELS, bias=False)
    if lora_args.use_lora:
        training_args.optim = "paged_adamw_8bit"
        training_args.fp16 = True
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=512,
        padding_side=PADDING_SIDE,
        truncation_side=PADDING_SIDE,
        padding=True,
        truncation=True
    )

    ###############################
    # 定义 LoRA 配置
    ###############################
    if lora_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            inference_mode=lora_args.inference_mode,
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if 'lm_head' in name:
                param.requires_grad = False
        if training_args.local_rank == 0:
            model.print_trainable_parameters()

    if custom_args.eval_only or custom_args.ckpt_path is not None:
        if custom_args.ckpt_path:
            # Load the model weights
            print('loading weights for evaluation')
            if not custom_args.ckpt_path.endswith('.safetensors'):
                model.load_state_dict(torch.load(custom_args.ckpt_path, map_location='cpu'), strict=True)
            else:
                loaded_weights = load_file(custom_args.ckpt_path)
                model.load_state_dict(loaded_weights, strict=False)
    
    ################################
    # 实例化DataCollatorWithPadding
    ################################
    print('loading PatternDataCollatorWithPadding')
    data_collator = PatternDataCollatorWithPadding(tokenizer=tokenizer)
    data_collator.padding_side = PADDING_SIDE
    ################################
    # 实例化 MoEPatternDataset
    ################################
    print('loading MoEPatternDataset')
    origin_data = load_dataset(data_args.data_path)['train']
    shuffled_data = origin_data.shuffle(seed=666)
    train_size = int(len(shuffled_data) * 0.9)
    train_data = shuffled_data.select(range(train_size))
    eval_data = shuffled_data.select(range(train_size, len(shuffled_data)))
    # train_data = origin_data
    # train_data = eval_data = origin_data
    train_dataset = MoEPatternDataset(
        train_data,
        training=True,
        train_max_seq_size=custom_args.train_max_seq_size,
        num_experts_per_layer=num_experts_per_layer
    )
    eval_dataset = MoEPatternDataset(
        eval_data,
        training=False,
        eval_max_seq_size=custom_args.eval_max_seq_size,
        num_experts_per_layer=num_experts_per_layer
    )

    # decouple optimization of base and head modules
    len_dataloader = len(train_dataset) / training_args.per_device_train_batch_size
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, training_args.num_train_epochs * len_dataloader,
        learning_rate_head=custom_args.lr_head,
        learning_rate_base=custom_args.lr_base,
        weight_decay=training_args.weight_decay
    )
    print("#params:", sum([p.numel() for p in model.parameters()]))
    print("trainable #params:", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    model = model.cuda()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )
    model.config.use_cache = False
    if custom_args.eval_only:
        print('Start evaluating')
        with torch.inference_mode():
            results = trainer.evaluate()
            print(results)
        return results

    print('Start training')
    # if list(pathlib.Path(output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    trainer.train()
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir+'/model_state_dict', state_dict=model.state_dict(), safe_serialization=False)


if __name__ == "__main__":
    train()
    # test_dataset()
    # test_model()

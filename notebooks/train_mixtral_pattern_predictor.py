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
    DataCollatorWithPadding,
    Trainer,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import CausalLMOutputWithPast


from peft import get_peft_model, LoraConfig

num_layers = 32
num_experts_per_layer = 8
NUM_LABELS = num_layers * num_experts_per_layer
PADDING_SIDE = 'left'
model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

def create_optimizer_and_scheduler(
    model, num_training_steps, learning_rate_head=2e-4, learning_rate_base=2e-5, warmup_steps=5000):
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
            "weight_decay": 5e-3,
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
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=f"marsggbo/bigbench4switch{num_experts_per_layer}_pattern_predictor", metadata={"help": "Path to the training data."}
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
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
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
    ipdb: bool = field(default=False, metadata={"help": "debug with ipdb"})


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
        # return 8
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_data = self.data[idx]
        input_ids = seq_data['token_ids'] # (seq_len,)
        input_ids = torch.tensor(input_ids, dtype=int)
        seq_len = len(input_ids)
        prompt_len = seq_data['prompt_tokens_len']
        decoding_len = seq_len - prompt_len
        labels = np.stack(seq_data['token_pattern_matrices']) # (seq_len, #layers, #experts)
        labels = torch.from_numpy(labels).int()
        num_to_pad = self.num_experts_per_layer - labels.shape[-1]
        pad_labels = torch.zeros(*labels.shape[:-1], num_to_pad)
        labels = torch.cat((labels, pad_labels), dim=-1)
        attention_mask = torch.ones(len(input_ids)) # (seq_len,)

        if self.training:
            self.truncate_ratio = random.uniform(0.1, 1)
            if self.train_max_seq_size is not None:
                truncate_length = min(int(decoding_len * self.truncate_ratio), self.train_max_seq_size)
            else:
                truncate_length = int(decoding_len * self.truncate_ratio)
        else:
            if self.eval_max_seq_size is not None:
                truncate_length = min(int(decoding_len * self.truncate_ratio), self.eval_max_seq_size)
            else:
                truncate_length = int(decoding_len * self.truncate_ratio)
        start_index = 0
        end_index = prompt_len + truncate_length
        return {
            'input_ids': input_ids[start_index:end_index],
            'attention_mask': attention_mask[start_index:end_index],
            'labels': labels[start_index:end_index],
            'decode_len': truncate_length
        }


@dataclass
class PatternDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        if not hasattr(self, 'padding_side'):
            self.padding_side = 'left'
        assert self.padding_side in ['left', 'right'], "Padding should be on one side (left or right)"
        non_label_features =[]
        for feature in features:
            item = {key: val for key, val in feature.items() if key in ['input_ids', 'attention_mask']}
            non_label_features.append(item)
        batch = super().__call__(non_label_features)
        label_features = [feature['labels'] for feature in features] # (seq_len, L*E) 可以是 0/1，也可是 weighted values
        # 计算最大长度以进行padding
        max_length = max(len(label) for label in label_features)
        # 进行padding：对于不足max_length的部分，用全0的pattern填充
        padded_labels = []
        for label in label_features:
            # 创建一个足够大的全 0 tensor来存放padded的labels
            label = torch.tensor(label)
            padded_label = torch.zeros((max_length, label.shape[-2], label.shape[-1]))
            # 将实际的label值复制到全0tensor的前面
            if self.padding_side == 'left':
                padded_label[-1*len(label):, :] = label # padding left
            else:
                padded_label[:len(label), :] = label # padding right
            padded_labels.append(padded_label)
        # 将padded_labels转换为一个tensor，并加入到batch中
        batch['labels'] = torch.stack(padded_labels, dim=0)
        batch['decode_len'] = torch.tensor([feature['decode_len'] for feature in features])
        
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


def new_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    decode_len: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # logits = []
        # for i in range(len(self.lm_head)):
        #     logits_per_expert = self.lm_head[i](hidden_states).float() # (bs, seq_len, num_experts)
        #     logits.append(logits_per_expert)
        # logits = torch.stack(logits, dim=-2) # # (bs, seq_len, num_layers, num_experts)
        logits = self.lm_head(hidden_states).float() # (bs, seq_len, num_layers*num_experts)

        loss = None
        if labels is not None:
            # masked BCE loss
            labels = labels.to(logits.device).float()
            # w1 = torch.zeros_like(labels)
            # pad_lens = (attention_mask==0).sum(-1).view(-1).int()
            # seq_lens = attention_mask.sum(-1).view(-1).int()
            # for i in range(labels.size(0)):
            #     # w1[:, pad_lens[i]:,:,:]=1
            #     # w1[:, pad_lens[i]:seq_lens[i]-decode_len[i],:,:]=0.2 # prompt token weights
            #     w1[:,-1*decode_len[i]:,:,:] = 1 # decoding token weights
            # loss_fn = torch.nn.BCEWithLogitsLoss(weight=w1, reduction='mean')
            # loss = loss_fn(logits.view(*labels.shape), labels)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1, num_experts_per_layer),
                labels.view(-1, num_experts_per_layer),
                reduction='none')
            loss_mask = labels.view(-1, num_experts_per_layer).sum(-1) != 0
            loss = loss[loss_mask].sum() / loss_mask.sum()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
    accuracy = TP / (num_tokens*64)
    print(f"non-padding ratio: {indices.sum()}/{len(indices)}={indices.sum()/len(indices)}\n")

    return {
        'accuracy': accuracy,
    }

def top_2_one_hot(logits):
    batch_size, seq_len, num_layer, num_experts = logits.shape
    
    # 获取top-2索引，values不使用
    _, top_2_indices = logits.topk(2, dim=-1, largest=True, sorted=True)  # shape: (batch_size, seq_len, num_layer, 2)
    
    # 初始化一个全0的tensor
    one_hot = torch.zeros_like(logits)  # shape: (batch_size, seq_len, num_layer, num_experts)
    
    # 生成one hot矩阵
    batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1).expand(-1, seq_len, num_layer, 2)
    seq_indices = torch.arange(seq_len).view(1, seq_len, 1, 1).expand(batch_size, -1, num_layer, 2)
    layer_indices = torch.arange(num_layer).view(1, 1, num_layer, 1).expand(batch_size, seq_len, -1, 2)
    expert_indices = top_2_indices  # 直接使用top-2索引
    
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


def compute_metrics(outputs, decode_len=64):
    true_labels = outputs.label_ids # (bs, seq_len, num_layer, num_experts)
    pred_labels = outputs.predictions # (bs, seq_len, num_layer*num_experts)
    bs, seq_len, num_layer, num_experts = true_labels.shape
    labels = true_labels.reshape(*true_labels.shape[:2], -1)
    mask = labels.mean(-1)!=-100
    valid_lens = mask.sum(-1)
    start_indices = valid_lens-decode_len
    all_indices = start_indices[:, np.newaxis] + np.arange(decode_len)
    true_labels = true_labels[np.arange(bs)[:, np.newaxis], all_indices, ...]
    pred_labels = pred_labels[np.arange(bs)[:, np.newaxis], all_indices, ...]
    if len(pred_labels.shape) == 3:
        bs, seq_len, dim = pred_labels.shape
    elif len(pred_labels.shape)==4:
        bs, seq_len, num_layer, num_experts = pred_labels.shape
        dim = num_layer * num_experts
    assert dim == NUM_LABELS, "Dimension of predictions should be {} but got {}".format(NUM_LABELS, dim)
    true_labels = true_labels.reshape(-1, num_experts_per_layer)
    pred_labels = pred_labels.reshape(-1, num_experts_per_layer)
        
    # Convert predictions to top-2 one-hot encoding
    preds_one_hot = np.zeros_like(pred_labels)
    top2_indices = np.argsort(pred_labels, axis=1)[:, -2:]
    rows = np.arange(pred_labels.shape[0])[:, None]
    preds_one_hot[rows, top2_indices] = 1

    return acc_precision_recall_f1(
        true_labels, preds_one_hot
    )

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
    config = AutoConfig.from_pretrained(
        model_name_or_path
    )
    # # for debug
    # config.num_hidden_layers = 8
    # config.hidden_size = 1024
    # config.intermediate_size = 2048
    # model = AutoModelForCausalLM.from_config(config)
    ## for real training
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir="/data/common/mixtral/")
    predictor_num_layers = 2
    model.model.layers = model.model.layers[:predictor_num_layers]
    model.forward = types.MethodType(new_forward, model)
    model.lm_head = nn.Linear(model.config.hidden_size, NUM_LABELS, bias=False)
    # model.lm_head = nn.ModuleList([
    #     nn.Linear(model.config.hidden_size, 8, bias=False) for i in range(32)
    # ])
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=512,
        padding_side=PADDING_SIDE,
        truncation_side=PADDING_SIDE,
        padding=True,
        truncation=True
    )
    ## 方案 1：设置 pad_token
    tokenizer.pad_token = tokenizer.unk_token
    
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
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.model.model.norm.weight.requires_grad = True
        for module in model.lm_head:
            module.weight.requires_grad = True
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
    origin_data = load_dataset("marsggbo/mixtral_8x7b_moe_alpaca_2k_token_pattern")['train']
    # shuffled_data = origin_data.shuffle(seed=666)
    # train_size = int(len(shuffled_data) * 0.9)
    # train_data = shuffled_data.select(range(train_size))
    # eval_data = shuffled_data.select(range(train_size, len(shuffled_data)))
    train_data = eval_data = origin_data
    train_dataset = MoEPatternDataset(
        train_data,
        training=True,
        train_max_seq_size=custom_args.train_max_seq_size
    )
    eval_dataset = MoEPatternDataset(
        eval_data,
        training=False,
        eval_max_seq_size=custom_args.eval_max_seq_size,
    )

    # decouple optimization of base and head modules
    len_dataloader = len(train_dataset) / training_args.per_device_train_batch_size
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, training_args.num_train_epochs * len_dataloader,
        learning_rate_head=custom_args.lr_head,
        learning_rate_base=custom_args.lr_base
    )
    print("#params:", sum([p.numel() for p in model.parameters()]))
    print("trainable #params:", sum([p.numel() for p in model.parameters() if p.requires_grad]))

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
        results = trainer.evaluate()
        print(results)
        return results

    print('Start training')
    output_dir = training_args.output_dir
    if training_args.run_name:
        output_dir += f'{training_args.run_name}'
        training_args.output_dir = output_dir
    if list(pathlib.Path(output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir+'/model_state_dict', state_dict=model.state_dict(), safe_serialization=False)


def test_dataset():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.unk_token
    data_collator = PatternDataCollatorWithPadding(tokenizer=tokenizer)
    dataset = MoEPatternDataset('./merged_data.pt', data_type=1)
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=data_collator) 
    for idx, batch in enumerate(data_loader):
        if idx > 5:
            break
        print("Batch input_ids shape:", batch["input_ids"].shape)  # 打印每个batch的input_ids的shape
        print("Batch attention_mask shape:", batch["attention_mask"].shape)  # 打印每个batch的attention_mask的shape
        print("Batch labels shape:", batch["labels"].shape)  # 打印每个batch的attention_mask的shape
        print('\n\n=========')

def test_model():
    config = AutoConfig.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")
    # for debug
    config.num_hidden_layers = 4
    config.hidden_size = 1024
    config.intermediate_size = 2048
    model = AutoModelForCausalLM.from_config(config)
    model.forward = types.MethodType(new_forward, model)
    model.lm_head = nn.Linear(model.config.hidden_size, NUM_LABELS, bias=False)
    lora_args = LoraArguments()
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        inference_mode=lora_args.inference_mode,
        task_type="CAUSAL_LM",
    )
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=field(
    #         default_factory=lambda: ["q_proj", "v_proj"]
    #     ),
    #     lora_dropout=0.05,
    #     bias="none",
    #     inference_mode=False,
    #     task_type="CAUSAL_LM",
    # )
    model = get_peft_model(model, lora_config)
    model.lm_head.weight.requires_grad = True
    model.print_trainable_parameters()
    for named, param in model.named_parameters():
        if param.requires_grad:
            print(named, param.shape)
    x = torch.randint(0, 100, (4, 64))
    labels = torch.ones((4, 64, NUM_LABELS))
    output = model(x, labels=labels)
    print(output.logits.shape)
    print(output.loss)

if __name__ == "__main__":
    train()
    # test_dataset()
    # test_model()

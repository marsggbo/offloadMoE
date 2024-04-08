import sys
from fastapi import FastAPI
from pydantic import BaseModel
sys.path.append("mixtral-offloading")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.build_model import OffloadConfig, QuantConfig, build_model

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

config = AutoConfig.from_pretrained(quantized_model_name)

device = torch.device("cuda:0")

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 3
# offload_per_layer = 5
###############################################################

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)

from transformers import TextStreamer


tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None

seq_len = 0

app = FastAPI()
class ChatInput(BaseModel):
  user_input: str
  output_len: int
class ChatOutput(BaseModel):
  response: int

@app.post("/chat",response_model = ChatOutput)
async def chat(input:ChatInput):
  user_input = input.user_input
  user_entry = dict(role="user", content=user_input)
  input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

  # if past_key_values is None:
  attention_mask = torch.ones_like(input_ids)
  # else:
  #   seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
  #   attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

  result = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=None,
    streamer=streamer,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    max_new_tokens=input.output_len,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_hidden_states=True,
  )

  sequence = result["sequences"]
  # decoded_text = tokenizer.decode(sequence[0], skip_special_tokens=True)
  token_count = len(sequence[0])
  return ChatOutput(response=token_count)
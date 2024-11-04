from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import random
from copy import deepcopy
import os
from datetime import datetime
import json
from tqdm import tqdm

def list1(l):
    sum =0
    for i in l:
        sum+=i
    return sum

def str1(s):
    sum = 0
    for c in s:
        sum+=int(c)
    return sum

def l2s(tmp):
    return ''.join(str(x) for x in tmp)

ENV = ""
if torch.cuda.is_available():
    ENV="cuda"
else:
    ENV="cpu"

# 原始模型和分词器的初始化
origin_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(ENV)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "en_XX"


decoder_forward_func = origin_model.model.decoder.layers[11].forward

random_number = random.uniform(-1, 1)
t = 1
lasttoken_pos = 0
last_token_state = torch.zeros(5, 1, 1024)



def new_decoder_activate(hidden_states, attention_mask = None, encoder_hidden_states = None, encoder_attention_mask = None,
                         layer_head_mask = None, cross_attn_layer_head_mask = None, past_key_value = None, output_attentions = False,
                         use_cache = True):
    global last_token_state
    modified_hidden_states = hidden_states  # 加随机数 torch.nn.functional.gelu
    last_token_state = modified_hidden_states
    # 调用原始 decoder 层的 forward 方法
    output = decoder_forward_func(
        modified_hidden_states, attention_mask, encoder_hidden_states,
        encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask,
        past_key_value, output_attentions, use_cache
    )
    # final_output = output * 2  # 举例，对输出进行缩放
    return output

activated_model = deepcopy(origin_model)

activated_model.model.decoder.layers[11].forward = new_decoder_activate

cluster0_encodes = []
with open('random_activate_output/activated_better_translation.json', 'r', encoding='utf-8') as file:
    cluster = json.load(file)
cluster0 = cluster
for item in cluster0:
    tmp_encoded_input = tokenizer(item, return_tensors="pt").to(ENV)
    cluster0_encodes.append(tmp_encoded_input)

cluster0_origin = torch.zeros(5, 1, 1024).to(ENV)
for index, encoded_input in tqdm(enumerate(cluster0_encodes), total=len(cluster0_encodes)):
    i=1
    while i > 0:
        t = 1
        generated_tokens = activated_model.generate(
            **encoded_input, 
            forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
        )
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        i-=1
    cluster0_origin += last_token_state



current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
with open(f'hook_original_output/original_hiddenstate.json', 'w', encoding='utf-8') as file:
    json.dump((cluster0_origin/len(cluster0_encodes)).tolist(), file, ensure_ascii=False, indent=4, separators=(',', ':'))

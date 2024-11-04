from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import random
from copy import deepcopy
import gc
import os
from datetime import datetime
import json
from get_fvec import SentenceParser

parser = SentenceParser()

def compute_list1(l):
    sum =0
    for i in l:
        sum+=i
    return sum

def compute_str1(s):
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
    global t
    global last_token_state
    if t % 4 == 0:
        random_number = random.uniform(-1, 1)
        modified_hidden_states = hidden_states
        modified_hidden_states += random_number
        modified_hidden_states = torch.nn.functional.gelu(modified_hidden_states)
    else:
        modified_hidden_states = hidden_states  # 加随机数 torch.nn.functional.gelu
    t+=1
    t%=4
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
with open('.\\processed\\02clustered_original_translation.json', 'r', encoding='utf-8') as file:
    cluster = json.load(file)
cluster0 = cluster['0']
for item in cluster0:
    tmp_encoded_input = tokenizer(item['en'], return_tensors="pt").to(ENV)
    cluster0_encodes.append(tmp_encoded_input)


#保存变好的翻译
translated_sens=[]
#计算变好的翻译句子数
improved_translation = 0
tmp_mean_hiddenstate = torch.zeros(5, 1, 1024).to(ENV)
for index, encoded_input in enumerate(cluster0_encodes):
    i=10
    zh_tmp = []
    fvec_list_tmp = []
    better = 0
    last_token_hidden_states = torch.zeros(5, 1, 1024).to(ENV)
    while i > 0:
        t = 1
        generated_tokens = activated_model.generate(
            **encoded_input, 
            forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
        )
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        translated_text = translated_text.replace(" ", "")
        parser.clear()
        fvc_tmp = parser.get_feature_vector(translated_text)
        fvc_tmp = parser.get_feature_vector_dep(translated_text)
        if compute_list1(fvc_tmp) < compute_str1(cluster0[index]['fvec']):
            better+=1
            zh_tmp.append(translated_text)
            last_token_hidden_states += last_token_state
            fvec_list_tmp.append(l2s(fvc_tmp))
        i-=1
    if better > 0:
        improved_translation += 1
        tmp_mean_hiddenstate += last_token_hidden_states / better
        translated_sens.append({'en': cluster0[index]['en'], 'baseline': cluster0[index]['zh'], 'zh': zh_tmp , 'fvec': fvec_list_tmp})

mean_hiddenstate = tmp_mean_hiddenstate / improved_translation

with open(f'random_activate_output/activated_better_translation.json', 'w', encoding='utf-8') as file:
    json.dump(translated_sens, file, ensure_ascii=False, indent=4, separators=(',', ':'))

with open('random_activate_output/cluster0_steer.json', 'w', encoding='utf-8') as file:
    json.dump(mean_hiddenstate.tolist(), file, ensure_ascii=False, indent=4, separators=(',', ':'))
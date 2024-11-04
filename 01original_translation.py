from datetime import datetime
import json
import os
import sys
import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def read_txt_files_in_directory(directory):
    result = []
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为 .txt 文件
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            print(f"正在读取文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    result.append(line.strip())  # 使用 strip() 去除每行末尾的换行符
    return result

if __name__ == '__main__':
    ENV = ""
    if torch.cuda.is_available():
        ENV="cuda"
    else:
        ENV="cpu"

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(ENV)
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "en_XX"

    arg1_path = sys.argv[1]
    en_sens = read_txt_files_in_directory(arg1_path)

    # 保存翻译结果
    translated_sens = []

    for en_sen in tqdm(en_sens, desc="Processing", ncols=100):
        encoded_input = tokenizer(en_sen, return_tensors="pt").to(ENV)
        generated_tokens = model.generate(
            **encoded_input, 
            forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
        )
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        translated_sens.append({"en": en_sen, "zh": translated_text, "fvec": []})


    # 当前时间
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # 保存翻译结果
    with open(f'processed/01original_translation.json', 'w', encoding='utf-8') as file:
        json.dump(translated_sens, file, ensure_ascii=False, indent=4, separators=(',', ':'))
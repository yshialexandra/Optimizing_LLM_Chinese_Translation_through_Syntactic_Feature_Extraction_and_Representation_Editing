import os
import json


def read_json_files_in_directory(directory):
    reslut = []
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为 .txt 文件
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            print(f"正在读取文件: {file_path}")
            
            # 逐行读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                file = json.load(file)
                for line in file:
                    if(len(line['zh'])>0):
                        reslut.append(line['en'])  # 使用 strip() 去除每行末尾的换行符
    return reslut

if __name__ == "__main__":
    result = read_json_files_in_directory('output')
    with open('cluster0_better.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
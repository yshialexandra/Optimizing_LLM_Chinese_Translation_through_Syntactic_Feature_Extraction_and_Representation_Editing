import os
import json

def list_txt_files(folder_path):
    result = []
    # 遍历文件夹及其子文件夹中的所有文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):  # 只处理以 .txt 结尾的文件
                file_path = os.path.join(root, file)
                result.append(file_path)  # 返回文件的完整路径
    return result
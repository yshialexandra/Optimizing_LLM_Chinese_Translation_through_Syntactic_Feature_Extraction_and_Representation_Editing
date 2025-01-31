import os

def count_chars_in_txt_files(folder_path):
    """统计文件夹下所有txt文件的总字符数（排除换行符和空格）"""
    total_chars = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        # 移除换行符和空格后统计字符
                        cleaned_content = content.replace("\n", "").replace(" ", "")
                        total_chars += len(cleaned_content)
                except UnicodeDecodeError:
                    print(f"⚠️ 解码失败: {filepath}（尝试其他编码？）")
                except Exception as e:
                    print(f"❌ 处理文件出错: {filepath} - {str(e)}")
    return total_chars

if __name__ == "__main__":
    target_folder = input("请输入文件夹路径: ").strip()
    if not os.path.isdir(target_folder):
        print("❌ 路径无效或不是文件夹")
    else:
        total = count_chars_in_txt_files(target_folder)
        print(f"\n📂 文件夹: {target_folder}")
        print(f"📝 总字符数: {total}（已排除换行符和空格）")
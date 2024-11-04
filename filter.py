import json
from datetime import datetime

def compute1(s):
    sum = 0
    for c in s:
        sum+=int(c)
    return sum

with open('.\\processed\\2024-09-14-21-42-57_processed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

result = []

for item in data:
    if compute1(item['fvec']) > 0:
        result.append(item)
print(len(result))

# 当前时间
now = datetime.now()
# 格式化时间
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")

with open(f'.\\processed\\{formatted_time}_processed.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
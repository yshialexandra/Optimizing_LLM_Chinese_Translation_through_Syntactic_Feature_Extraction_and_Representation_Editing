import json
from get_fvec import SentenceParser

parser = SentenceParser()

with open ('processed/01original_translation.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    original_translation = [i['zh']for i in data[0]]

for index, i in enumerate(original_translation[0]):
    parser.clear()
    i_fvec = parser.write_constituency_vector(i)
    i_fvec = parser.write_dependency_vector(i)
    str_tmp = ''.join(str(x) for x in i_fvec)
    data[index]["fvec"] = str_tmp

with open('processed/01original_translation.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

import json

dataset_path = "Data/dev.json"
dataset_cnt = 10
data = []

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
    cnt = 0
    for d in dataset:
        if cnt == dataset_cnt:
            break
        data.append([
            d[0],
            {"sentences": d[1]["sentences"]}
        ])
        cnt += 1
        
with open("Data/dev/test.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print("dumped {} events".format(len(data)))
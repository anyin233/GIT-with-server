import json

dataset_path = "Data/dev.json"
data = []

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
    for d in dataset:
        data.append({
            "id": d[0],
            "eventspan": d[1]["sentences"],
            "entity": d[1]["ann_mspan2guess_field"],
            "event_list": d[1]["recguid_eventname_eventdict_list"]
        })
        
with open("Data/readable/dev.json", 'w', encoding='utf-8') as f:
    json.dump(data, f)

print("dumped {} events".format(len(data)))
print("\n".join(data[0]["eventspan"]))
import pickle
import dee

pkl_dir = "Exps/try/Output/dee_eval.test.gold_span.GIT.12.pkl"
out_dir = "helper_data/data_single.txt"

with open(pkl_dir, 'rb') as f:
    content = pickle.load(f)
    print(type(content[0]))
    print(content[0])
    with open(out_dir, 'w') as f:
        f.write(str(content[0]))
    
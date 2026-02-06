import pickle
import os

pkl_path = "/home/2022113135/gyucheol/NetfLips/av2av-main/assets/samples/en/TRajLqEaWhQ_00002.bbox.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"Type of data: {type(data)}")
if isinstance(data, list):
    print(f"Length of list: {len(data)}")
    if len(data) > 0:
        print(f"First element: {data[0]}")
elif isinstance(data, dict):
    print(f"Keys: {data.keys()}")
    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            print(f"{k} length: {len(v)}")
            if len(v) > 0:
                print(f"{k} first element: {v[0]}")
        else:
            print(f"{k}: {v}")
else:
    print(data)

import os

for split in ["train", "val", "test"]:
    print(split)
    for cls in os.listdir(f"dataset/{split}"):
        print(cls, len(os.listdir(f"dataset/{split}/{cls}")))
    print("-" * 25)

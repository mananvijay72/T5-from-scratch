import os
import random

def split_dataset(input_file, train_file, val_file, test_file, split=(0.8,0.1,0.1)):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)
    n = len(lines)
    n_train = int(split[0]*n)
    n_val   = int(split[1]*n)

    with open(train_file, "w", encoding="utf-8") as f: f.writelines(lines[:n_train])
    with open(val_file, "w", encoding="utf-8") as f: f.writelines(lines[n_train:n_train+n_val])
    with open(test_file, "w", encoding="utf-8") as f: f.writelines(lines[n_train+n_val:])

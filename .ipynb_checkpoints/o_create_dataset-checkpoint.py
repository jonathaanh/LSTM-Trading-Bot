import pandas as pd
import numpy as np
import torch

GRAB = ''
SAVE = ''
DATASET_SIZE = 48

def read_in(path):
    return pd.read_pickle(path)

def slice_jawn(df, size):
    df_len = len(df)
    set_stack = []
    for i in range(df_len):
        set_stack.append(df.iloc[i: i + size].T)
    return set_stack
        
        

def do_it(grab_path=GRAB, save_path=SAVE, size = DATASET_SIZE, cols = []):
    df = torch.tensor(read_in(grab_path)[cols].to_numpy())
    torch_stack = torch.stack(slice_jawn(df, size))
    pd.to_pickle(df, filepath=save_path)
    
do_it()
import os
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import forgi.visual.mplotlib as fvm
import forgi

symbols = ('.', '(', ')', '[', ']', '{', '}', '<', '>')
symb_voc = {
     '.':0, 
     '(':1,
     ')':2, 
     '[':1, 
     ']':2, 
     '{':1, 
     '}':2, 
     '<':1, 
     '>':2,
    }
trg_pad_idx = 4

def get_seq(filename, structure='both'):
    '''
    filename.dot
    '''
    with open(filename) as f:
        lines = f.readlines()
    if structure == 'secondary':
        return lines[-1][:-1]
    elif structure == 'primary':
        return lines[-2][:-1]
    elif structure == 'both':
        return lines[-2][:-1], lines[-1][:-1]


def get_data(data_dir):
    src = []
    trg = []
    for filename in tqdm(os.listdir(data_dir)):
        seq1, seq2 = get_seq(data_dir/filename, structure = 'both')
        src.append(seq1)
        trg.append(seq2)
    return src, trg


def get_df(src, trg):

   
    df = pd.DataFrame(columns = ['src', 'trg'])
    df['src'] = src
    df['trg'] = trg
    df['src'] = [[char for char in df['src'][k]] for k in range(len(df.src))]
    df['trg'] = [[char for char in df['trg'][k]] for k in range(len(df.trg))]
    df['trg'] = [[symb_voc[char] if char in symbols else 0 for char in df['trg'][k]] for k in range(len(df.trg))]
    return df

def get_predictions(iterator, model):
    true_seq = []
    pred_seq = []
    prim_str = []
    for i, batch in enumerate(iterator):
        for line in range(len(batch)):
            true_sec_str = batch.trg[line].numpy()
            try:
                ind = true_sec_str.tolist().index(trg_pad_idx) 
            except ValueError:
                ind = None
            true_sec_str = true_sec_str.tolist()[:ind]
            pr_str = batch.src[line].numpy().tolist()[:ind]

            pred = model(batch.src[line].unsqueeze(0))
            probs = F.softmax(pred, dim = 2)
            results = torch.argmax(probs, dim=2)
            results = results.squeeze().detach().numpy().tolist()[:ind]

            prim_str.append(pr_str)
            true_seq.append(true_sec_str)
            pred_seq.append(results)
            
    result_df = pd.DataFrame()
    
    result_df['primary'] = prim_str
    result_df['secondary_true'] = true_seq
    result_df['secondary_pred'] = pred_seq
    
    return result_df

def ind2seq(df):
    result_df = df.copy()
    result_df['sec_true_db'] = [[symbols[ind] for ind in result_df['secondary_true'][k]] for k in range(len(result_df['secondary_true']))]
    result_df['sec_true_db'] = [str.join('',result_df['sec_true_db'][k])for k in range(len(result_df['sec_true_db']))]
    result_df['sec_pred_db'] = [[symbols[ind] for ind in result_df['secondary_pred'][k]] for k in range(len(result_df['secondary_pred']))]
    result_df['sec_pred_db'] = [str.join('',result_df['sec_pred_db'][k])for k in range(len(result_df['sec_pred_db']))]
    return result_df

def balance_op_tmp(string):
    stack = [] 
    spl_str = [ch for ch in string]  
        
    close_stack = []
    for i, char in enumerate(string): 
        if char == '.':
            pass
        elif char == "(": 
              stack.append(char) 
        else: 
            close_stack.append(char)
            if stack: 
                current_char = stack.pop() 

                if current_char == '(': 
                    if char != ")": 
                        stack.append(current_char) 
                    else:
                        close_stack.pop()

    if stack or close_stack: 
        for s in stack:
            idx = spl_str.index(s)
            spl_str[idx]='.'
        for s in close_stack:
            tmp = spl_str[::-1]
            idx = tmp.index(s)
            tmp[idx]='.'  
            spl_str = tmp[::-1]  
    return ''.join(spl_str)
    

def visualize(prim_str, pred_string, true_string, save_imgs = False, suffix_img = ''):
    corr_string = balance_op_tmp(pred_string)

    print(f'pred: {pred_string}')
    print(f"true: {true_string}")
    print(f'corr: {corr_string}')

    with open('tmp_vis_pred.txt', 'w') as f:
        f.write(prim_str+ os.linesep)
        f.write(corr_string+ os.linesep)
        f.close
    with open('tmp_vis_true.txt', 'w') as f:
        f.write(prim_str+ os.linesep)
        f.write(true_string+os.linesep)
        f.close

    plt.figure(figsize = (20, 20))
    plt.title('predicted')
    cg = forgi.load_rna('tmp_vis_pred.txt', allow_many=False)
    fvm.plot_rna(cg, text_kwargs={"fontweight":"black"}, lighten=0.7,backbone_kwargs={"linewidth":3})
    if save_imgs:
        plt.savefig(f'pred_{suffix_img}.jpg')
    plt.show()

    plt.figure(figsize = (20, 20))
    plt.title('original')
    cg = forgi.load_rna('tmp_vis_true.txt', allow_many=False)
    fvm.plot_rna(cg, text_kwargs={"fontweight":"black"}, lighten=0.7,backbone_kwargs={"linewidth":3})
    plt.savefig(f'true_{suffix_img}.jpg')
    plt.show()
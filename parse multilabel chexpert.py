#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import pickle
import re
import numpy as np
import ast
from sklearn.metrics import f1_score

test_frame = pd.read_csv('C:/Users/sonne/Documents/Github/biasICL/chexpert_binaryPNA_test_df_labels.csv', index_col=0)


# In[11]:


def parse_answers(text):
    pattern = r"<ANS>\s*([A-N, ]+)\s*</ANS>"
    match = re.search(pattern, text)
    letter_to_index = {chr(i): i - ord('A') for i in range(ord('A'), ord('N') + 1)}
    vector = [0] * 14
    if match:
        letters = match.group(1).split(',')
        for letter in letters:
            letter = letter.strip()
            if letter in letter_to_index:
                vector[letter_to_index[letter]] = 1
    return vector

def pickle_to_res(path):
    with open(path, "rb") as f:
        raw_pickle = pickle.load(f)

    results = {}
    answer_prefix="Answer "
    
    import re

    def extract_ans(ans_str, search_substring):
        # Split the string into lines
        lines = ans_str.split("\n")
        pattern = search_substring

        for line in lines:
            # Use regex to find the pattern
            match = re.search(pattern, line)
            if match:
                return parse_answers(line)
            
        return "ERROR"  # Answer not found

    for k, v in raw_pickle.items():
        if k != 'token_usage':  # Skip token_usage
            qns_idx = ast.literal_eval(k)
            for idx, qn_idx in enumerate(qns_idx):
                results[qn_idx] = extract_ans(
                    v[0], f"{answer_prefix}{idx+1}:"
                )  # We start with question 1
    
    return results


# In[48]:


def res_to_vec(results):
    test_df = pd.read_csv('chexpert_binaryPNA_test_df_labels.csv', index_col=0)
    num_errors = 0
    labels, preds, sex = [], [], []
    for i in test_df.itertuples():
        if (i.Index not in results):
            num_errors += 1

            print(i.Index, f"answer not found")
            continue

        pred_text = np.asarray(results[i.Index])
        ground_truth = (np.asarray(i[6:20]) == 1).astype(float)
        labels.append(ground_truth)
        preds.append(pred_text)
        sex.append(i.Sex)
        
    print(f"In total {num_errors} errors len = {len(labels)}")
    return labels,preds,sex


# In[66]:


labels,preds,sex = res_to_vec(pickle_to_res("C:/Users/sonne/Downloads/chexpertMultilabel_2_2_gpt-4o-2024-05-13_50.pkl"))


# In[67]:


print(f1_score(np.vstack(labels),np.vstack(preds),average='macro'))


# In[ ]:





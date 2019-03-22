import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from operator import itemgetter
import math
from sklearn.model_selection import train_test_split
import random
from random import choices

  
def I(df, I_type):
    if I_type == 'misscl':
        if df.empty:
            return 0
        p = (df.label == 1).sum()/df.shape[0]
        return min(p,1-p)
    if I_type == 'entropy':
        if df.empty:
            return 0
        p = (df.label == 1).sum()/df.shape[0]
        if p in [0,1]:
            return 0
        else:
            return -p*math.log2(p)-(1-p)*math.log2(1-p)
    if I_type == 'gini':
        if df.empty:
            return 0
        p = (df.label == 1).sum()/df.shape[0]
        return 2*p*(1-p)
    

def separate_train(node, I_type, n_row, l_col_sam):
    name = node[0]
    x = node[1]
    IG_arr = [] 
    IG_1 = x.shape[0]/n_row*I(x, I_type)
    for col in l_col_sam:
        for a in list(set(x[col])):
            IG = IG_1-x[x[col]<a].shape[0]/n_row*I(x[x[col]<a], I_type)-x[x[col]>=a].shape[0]/n_row*I(x[x[col]>=a], I_type)
            IG_arr.append([col, a, IG])
    col, a = max(IG_arr, key=itemgetter(2))[0], max(IG_arr, key=itemgetter(2))[1]
    node_l = [name+[0], x[x[col]<a]]
    node_r = [name+[1], x[x[col]>=a]]
    return node_l, node_r, {'route':name, 'variable':col, 'value':a, 'proportion_of_rows':x.shape[0]/n_row,'proportion_of_1':(x.label == 1).sum()/x.shape[0]}

    
def make_tree(df, I_type, depth, var_proportion):
    n_row = df.shape[0]
    df_repl = df.sample(n = df.shape[0], replace=True).reset_index(drop=True)
    l_cal = list(df.drop('label', axis=1).columns.values)
    l_col_sam = random.sample(l_cal, int(len(l_cal)*var_proportion))
    tree = [[[], df_repl]]
    leaves = []
    rules = []
    while len(tree)>0:
        node = tree[0]
        df = node[1]
        df_n = df.shape[0]
        leaves.append({'route': node[0], 'proportion_of_rows': df_n/n_row, 'proportion_of_1': (df.label == 1).sum()/df_n})
        if ((df.label == 1).sum() == df_n) | ((df.label == 0).sum() == df_n) | (len(node[0])==depth):
            tree.pop(0)
        else:
            node_l, node_r, rule = separate_train(node, I_type, n_row, l_col_sam)
            rules.append(rule)
            tree.pop(0)
            tree.append(node_l)
            tree.append(node_r)
    return leaves, rules


def separate_val(node, rules):
    name = node[0]
    x = node[1]
    for r in rules:
        if name == r['route']:
            col = r['variable']
            a = r['value']
            node_l = [name+[0], x[x[col]<a]]
            node_r = [name+[1], x[x[col]>=a]]
            return node_l, node_r
            
    
def tree_impl(df, tree, depth):
    tree_nodes = [[[], df]]
    rules = tree[1]
    leaves = tree[0]
    arr_df_prediction = []
    while len(tree_nodes)>0:
        node = tree_nodes[0]
        if len(node[0]) >= depth:
            r = pd.DataFrame(node[1]['label'])
            r['probability'] = [element['proportion_of_1'] for element in leaves if element['route'] == node[0]][0]
            arr_df_prediction.append(r[['probability']])    
            tree_nodes.pop(0)
        elif node[0] not in list(item['route'] for item in rules):
            r = pd.DataFrame(node[1]['label'])
            r['probability'] = [element['proportion_of_1'] for element in leaves if element['route'] == node[0]][0]
            arr_df_prediction.append(r[['probability']])
            tree_nodes.pop(0)
        else:
            node_l, node_r = separate_val(node, rules)
            tree_nodes.append(node_l)
            tree_nodes.append(node_r)
            tree_nodes.pop(0)
            
    #print(r)
    return pd.concat(arr_df_prediction, axis=0)#df_prediction


def auc(df, n):
    df['mean'] = df.drop(columns=['label']).iloc[:,list(range(0,n))].mean(axis=1)

    df = df.loc[:, ['label', 'mean']]
    R=[]
    for xx in sorted(list(set(df['mean']))):
        df.loc[df['mean'] < xx, 'leaf'] = 0 
        df.loc[df['mean'] >= xx, 'leaf'] = 1
        df.loc[((df['label']==1) & (df['leaf']==1)), 'res'] = 'TP'
        df.loc[((df['label']==0) & (df['leaf']==1)), 'res'] = 'FP'
        df.loc[((df['label']==1) & (df['leaf']==0)), 'res'] = 'FN'
        df.loc[((df['label']==0) & (df['leaf']==0)), 'res'] = 'TN'
        TP = (df.res == 'TP').sum()
        FP = (df.res == 'FP').sum()
        FN = (df.res == 'FN').sum()
        TN = (df.res == 'TN').sum()
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        R.append([FPR, TPR])
    R = sorted(R, key=itemgetter(0,1))
    auc = np.trapz(list(zip(*R))[1], x = list(zip(*R))[0])
    #print('ROC AUC =', auc)
    return auc, R
import glob
import os
import numpy as np
import pandas as pd

def merge_dict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    for key, value in dict1.items():
        dict1[key].extend(dict2[key])
    return dict1

def merge_multiple_dicts(list_of_dicts):
    res = {
        'precision': [],
        'recall': [],
        'accuracy': [],
        'fscore': [],
    }
    for d in list_of_dicts:
        merge_dict(res,d)
    return res


model_name = 'lcp_r2000_ConvPoint_SearchQuantized_archLrg_run4_SGD_CyclicLR_weights1,2_FocalLoss'
# model_name = 'lcp_r2000_ConvPoint_SearchQuantized_archLrg_run2_SGD_CyclicLR_weights1,2_FocalLoss'
# model_name = 'lcp_r2000_ConvPoint_SearchQuantized_archLrg_run3_SGD_CyclicLR_weights1,2_CrossEntropy'
# model_name = 'lcp_r5000_ConvPoint_SearchQuantized_arch2048_run2_Adam_StepLR_weights1,2_CrossEntropy'
# model_name = 'lcp_r5000_ConvPoint_SearchQuantized_archLrg-noFstConv_run2_Adam_StepLR_weights1,2_CrossEntropy'
# model_name = 'lcp_r1000_ConvPoint_SearchQuantized_archLrg-noFstConv_Adam_StepLR_weights1,2_CrossEntropy'
# files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/5000/arch2048/quantitative/testSet_context*.csv')
# files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/_Adam_StepLR_CrossEntropy/quantitative/testSet_context*.csv')
# files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/_Adam_StepLR_FocalLoss/quantitative/testSet_context*.csv')
# files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/5000/archLrg/quantitative/testSet_context*.csv')
files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_FocalLoss/quantitative/new/testSet_context*.csv')
# files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/_Adam_StepLR_FocalLoss/quantitative/new/testSet_context*.csv')
# files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/1000/archLrg/focus_fix/testSet_context*.csv')

print(f'Number of dicts: {len(files)}')

dicts=[]

for file in files:
    df = pd.read_csv(file)
    dicts.append(df.to_dict(orient='dict'))

dii=[]
for dic in dicts:
    d = {dic['Unnamed: 0'][0]:[dic['0'][0]], dic['Unnamed: 0'][1]:[dic['0'][1]], dic['Unnamed: 0'][2]:[dic['0'][2]], dic['Unnamed: 0'][3]:[dic['0'][3]]}
    dii.append(d)
# print(f'random dict {dii}')

result = merge_multiple_dicts(dii)

precisions = result['precision']
recalls = result['recall']
accuracies = result['accuracy']
fscores = result['fscore']

result = {
    'model name': model_name,
    'precision': np.mean(precisions),
    'recall': np.mean(recalls),
    'accuracy': np.mean(accuracies),
    'fscore': np.mean(fscores),
}

print(f'result: {result}')

df = pd.DataFrame.from_dict(result, orient='index')
# csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/5000/archLrg-noFstConv/quantitative/{model_name}_testSet_context_focus_fix_result.csv'
# csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/_Adam_StepLR_FocalLoss/quantitative/new/{model_name}_testSet_context_focus_fix_result.csv'
csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_FocalLoss/quantitative/new/{model_name}_testSet_context_focus_fix_result.csv'
# csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/5000/archLrg/quantitative/{model_name}_testSet_context_focus_fix_result.csv'
# csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_CrossEntropy/quantitative/new/{model_name}_testSet_context_focus_fix_result.csv'
# csv_

df.to_csv(csv_path)
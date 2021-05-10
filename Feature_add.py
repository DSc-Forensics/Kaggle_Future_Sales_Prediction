# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:53:25 2021

@author: PC
"""

import numpy as np
import pandas as pd
import os
import time
import gc

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales")

#Change version according to monthly aggregation method used
all_data=pd.read_parquet('Inputs/Base_train_v2.gzip')
shift_range = [12,9,6,3,2,1]
index_cols = ['shop_id', 'item_id', 'date_block_num']

def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range] or col[-2:] in [str(item) for item in shift_range]]
"""
month_shift=6
start=time.time()
cols_to_rename=[col for col in all_data.columns if '_'.join(col.split('_')[-2:])=='roll6_mean']
na_cols=[]

train_shift = all_data[index_cols + cols_to_rename].copy()    
train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
all_data = all_data[all_data['date_block_num'] >=18]    
foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
train_shift = train_shift.rename(columns=foo)
all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

print(str(month_shift)+" lags complete")
del train_shift
gc.collect()

for col in fit_cols:
    if '_'.join(col.split('_')[:-1])+'_roll12_mean' in all_data.columns:
        continue
    all_data['_'.join(col.split('_')[:-1])+'_roll12_mean']=all_data['_'.join(col.split('_')[:-1])+'_roll6_mean']+\
    all_data['_'.join(col.split('_')[:-1])+'_roll6_mean_lag_6']/2.0
    #na_cols.append('_'.join(col.split('_')[:-1])+'_roll3_mean_lag_3')
    na_cols.append('_'.join(col.split('_')[:-1])+'_roll6_mean')
"""    
all_data['target_ratio_1_2']=all_data['target_lag_1']/all_data['target_lag_2']
all_data['target_ratio_2_3']=all_data['target_lag_2']/all_data['target_lag_3']
rat_cols=['target_ratio_1_2','target_ratio_2_3']

drop_cols=[]
for col in all_data.columns:
    if col.endswith('roll3_mean_lag_3') or col.endswith('roll6_mean_lag_6'):
        drop_cols.append(col)
        
all_data=all_data.drop(drop_cols,axis=1)
gc.collect()

mean_cols=[]
for col in all_data.columns:
    if col.endswith('roll3_mean') or col.endswith('roll6_mean') or col.endswith('roll12_mean'):
        mean_cols.append(col)

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range] or col[-2:] in [str(item) for item in shift_range]]
fit_cols+=rat_cols
fit_cols+=mean_cols

to_drop_cols = list(set(list(all_data.columns))-(set(fit_cols)|set(['shop_id', 'date_block_num','item_category_id',\
            'city','city_size','shop_type','hdays','mdays','Broad_cat','target'])))

all_data.replace([np.inf, -np.inf], np.nan, inplace=True)

all_data[fit_cols]=all_data[fit_cols].replace(0,np.nan)

for col in fit_cols:
    if 'dateblock' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num'])[col].transform('max'),inplace=True)
    elif 'shoptypecat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_type','item_category_id'])[col].transform('max'),inplace=True)
    elif 'shoptype' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_type'])[col].transform('max'),inplace=True)
    elif 'shopbroadcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_id','Broad_cat'])[col].transform('max'),inplace=True)
    elif 'shopcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_id','item_category_id'])[col].transform('max'),inplace=True)
    elif 'citybroadcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','city','Broad_cat'])[col].transform('max'),inplace=True)
    elif 'citycat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','city','item_category_id'])[col].transform('max'),inplace=True)
    elif 'city' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','city'])[col].transform('max'),inplace=True)    
       
    
    elif 'shop' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_id'])[col].transform('max'),inplace=True)
    elif 'broadcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','Broad_cat'])[col].transform('max'),inplace=True)
    elif 'cat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','item_category_id'])[col].transform('max'),inplace=True)
    #elif 'all' in col:
        #all_data[col].fillna(all_data.groupby(index_cols)[col].transform('max'),inplace=True)
    elif 'broaditem' in col and col!='Trans_item_lag_1':
        all_data[col].fillna(all_data.groupby(['date_block_num','Broad_item'])[col].transform('max'),inplace=True) 
    elif 'item' in col and col!='Trans_item_lag_1':
        all_data[col].fillna(all_data.groupby(['date_block_num','item_id'])[col].transform('max'),inplace=True)  

all_data[fit_cols]=all_data[fit_cols].replace(np.nan,0)

all_data['seasonal']=12-all_data['date_block_num']%12
all_data = downcast_dtypes(all_data)

all_data=all_data.drop(to_drop_cols,axis=1)

all_data['city']=all_data['city'].astype('category')
all_data['city']=all_data['city'].cat.codes

all_data['shop_type']=all_data['shop_type'].astype('category')
all_data['shop_type']=all_data['shop_type'].cat.codes

all_data['Broad_cat']=all_data['Broad_cat'].astype('category')
all_data['Broad_cat']=all_data['Broad_cat'].cat.codes

del all_data['shop_type']
#Change version according to monthly aggregation method used
all_data.to_parquet('Inputs/Train_ready_v2.gzip',compression='gzip')

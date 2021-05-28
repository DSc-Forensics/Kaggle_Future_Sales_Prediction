# -*- coding: utf-8 -*-
"""
Created on Sat May 15 17:28:01 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import scipy.sparse 
from sklearn.metrics import r2_score

import gc
from sklearn.metrics import mean_squared_error as mse

from statsmodels.regression import linear_model

import warnings
warnings.filterwarnings("ignore")

import time

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales/Kaggle_Future_Sales_Prediction")

from catboost_CV import run_cat
from catboost_CV import plot_feature_importance
from lgbm_CV import run_lgbm
from xgboost_CV import run_xg
#from linreg_cv import run_regr

os.chdir("..")

sales=pd.read_hdf("Inputs/Total_shop_cat_translated_v2.hdf",key='df')

sales_by_item_id = sales[sales['date_block_num']<34].pivot_table(index=['item_id'],values=['item_cnt_day'], 
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'

del sales
gc.collect()

#Change version according to monthly aggregation method used
all_data=pd.read_parquet('Inputs/Train_ready_v3.gzip')
item_infos=pd.read_parquet('Inputs/Item_infos.gzip')

#del all_data['mths_since_shopitem_first']

num_outdated=6
all_data['Pred_target']=0.0

mods=['Cat','LGB','XG']
count=0
for model in [run_cat,run_lgbm,run_xg]:
    out=model(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=28,train_speed='slow')
    out.to_parquet('Train_ensemble_out/Best_newfts_'+mods[count]+'.gzip',compression='gzip')
    if model==run_xg:
        out=model(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=28,train_speed='slow',onehot=True)
        out.to_parquet('Train_ensemble_out/Best_newfts_'+mods[count]+'_onehot.gzip',compression='gzip')
    count+=1
    del out
    gc.collect()
    
all_data=pd.read_parquet('Inputs/Train_ready_v2.gzip')
#item_infos=pd.read_parquet('Inputs/Item_infos.gzip')

num_outdated=6
all_data['Pred_target']=0.0

mods=['Cat','LGB','XG']
count=0
for model in [run_cat,run_lgbm,run_xg]:
    out=model(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=28,train_speed='fast')
    out.to_parquet('Train_ensemble_out/Best_'+mods[count]+'_v2.gzip',compression='gzip')
    if model==run_xg:
        out=model(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=28,train_speed='fast',onehot=True)
        out.to_parquet('Train_ensemble_out/Best_'+mods[count]+'_onehot_v2.gzip',compression='gzip')
    count+=1
    del out
    gc.collect()

"""    
model = LinearRegression(n_jobs=-1).fit(x, y)
r_sq = model.score(x, y)
y_pred = model.predict(x)
"""
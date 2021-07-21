# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:05:07 2021

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

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales/Kaggle_Future_Sales_Prediction/Ensembling")

from catboost_CV import run_cat
from catboost_CV import plot_feature_importance
from lgbm_CV import run_lgbm
from xgboost_CV import run_xg
#from linreg_cv import run_regr

os.chdir("../..")

sales=pd.read_hdf("Inputs/Total_shop_cat_translated_v2.hdf",key='df')

sales_by_item_id = sales[sales['date_block_num']<34].pivot_table(index=['item_id'],values=['item_cnt_day'], 
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'

del sales
gc.collect()

#Change version according to monthly aggregation method used

item_infos=pd.read_parquet('Inputs/Item_infos.gzip')

all_data = pd.read_pickle("Inputs/Base_train2_69inc.pkl")
all_data.drop(['item_id','shop_id'],axis=1,inplace=True)
all_data['Pred_target']=0.0

last_block=34
niter=100
lr=0.04

city_cols=[]
for col in all_data.columns:
    if 'city' in col:
        city_cols.append(col)
        
all_data.drop(columns=city_cols,inplace=True)

for past_out in [6]:
    run_lgbm(all_data.copy(),sales_by_item_id,item_infos,st_mth=28,end_mth=34,num_outdated=past_out,return_data=False,\
    niter=niter,lrate=lr,cat_cols=['item_category_id','Broad_cat','supercategory_id','platform_id'\
                        ,'seasonal'])

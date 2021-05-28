# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:32:51 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
import scipy.sparse 
from sklearn.metrics import r2_score

import gc
from sklearn.metrics import mean_squared_error as mse

from statsmodels.regression import linear_model

import warnings
warnings.filterwarnings("ignore")

import time

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales/Train_ensemble_out")

for file in os.listdir():
    data=pd.read_parquet(file)
    data=data[data['date_block_num']<28]
    data.to_parquet(file,compression='gzip')
    
def run_regr(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=34,num_outdated=6,train_speed='fast',plot=False):

    if train_speed=='fast':
        lrate=0.04
        niter=100
    elif train_speed=='slow':
        lrate=0.004
        niter=1000
    else:
        lrate=0.0004
        niter=5000
        
    
    for last_block in range(st_mth,end_mth):
        #last_block = 33
        print(last_block)
        
        X_train = all_data.loc[all_data['date_block_num'] <  last_block].drop(['target','Pred_target'], axis=1)
        X_test =  all_data.loc[all_data['date_block_num'] ==  last_block].drop(['target','Pred_target'], axis=1)
        
        y_train = all_data.loc[all_data['date_block_num'] <  last_block, 'target'].clip(0,20).values
        y_test =  all_data.loc[all_data['date_block_num'] ==  last_block, 'target'].clip(0,20).values
        
        model = LinearRegression(n_jobs=-1).fit(X_train, y_train)
        r_sq = model.score(X_train, y_train)
        print("R-Squared Train LR : "+str(r_sq))
        
        pred_lgb = model.predict(X_test)
        pred_train=model.predict(X_train)
        
        pred_lgb=pred_lgb.clip(0,20)
        pred_train=pred_train.clip(0,20)
        
        del X_train,X_test
        gc.collect()
        
        outdated_items = sales_by_item_id[sales_by_item_id.loc[:,str(last_block-num_outdated):str(last_block-1)].sum(axis=1)==0]
        outdated_items = outdated_items[outdated_items.loc[:,'0':str(last_block-1)].sum(axis=1)>0]
        
        test=item_infos.loc[item_infos['date_block_num']==last_block]
        test['item_cnt_month']=pred_lgb
        test.loc[test['item_id'].isin(outdated_items['item_id']),'item_cnt_month']=0.0
        pred_lgb=test['item_cnt_month']
        
        print('R-squared for TRAIN - %f' % r2_score(y_train, pred_train))
        print('RMSE for TRAIN - %f' % mse(y_train, pred_train,squared=False))
        print('R-squared for CV - %f' % r2_score(y_test, pred_lgb))
        print('RMSE for CV - %f' % mse(y_test, pred_lgb,squared=False))
        
        all_data.loc[all_data['date_block_num'] == last_block,'Pred_target']=pred_lgb.values
        del y_train,y_test
            
    return all_data.loc[all_data['date_block_num']>=st_mth]
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:32:51 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import scipy.sparse 
from sklearn.metrics import r2_score

import gc
from sklearn.metrics import mean_squared_error as mse

from statsmodels.regression import linear_model

import warnings
warnings.filterwarnings("ignore")

import time

def run_lgbm(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=34,onehot=False,num_outdated=6,train_speed='fast',plot=False):
    
    if onehot:
        print("No onehot to work here")
        
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
        
        lgb_params = {'feature_fraction': 0.75,
                       'metric': 'rmse',
                       'nthread':8, 
                       'min_data_in_leaf': 20, 
                       'bagging_fraction': 0.75, 
                       'learning_rate': lrate, 
                       'objective': 'mse', 
                       'bagging_seed': 2**7, 
                       'num_leaves': 2**9,
                       'bagging_freq':1,
                       'verbose':0,
                       'lambda_l2':3,
                       'num_iterations':niter,
                       'categorical_feature':['name:item_category_id','name:shop_id','name:Broad_cat','name:city']}
        
        model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train,categorical_feature=['item_category_id',\
        'shop_id','city','Broad_cat']), niter)
        
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
        
        if plot==True:
            lgb.plot_importance(model,max_num_features=30)
            lgb.plot_importance(model,max_num_features=30,importance_type='gain')
            
    return all_data.loc[(all_data['date_block_num']>=st_mth)&(all_data['date_block_num']<end_mth)]
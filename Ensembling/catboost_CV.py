
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:32:51 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import scipy.sparse 
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

import gc
from sklearn.metrics import mean_squared_error as mse

from statsmodels.regression import linear_model

import warnings
warnings.filterwarnings("ignore")

import time
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df=fi_df[:20]
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

def run_cat(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=34,train_speed='fast',onehot=False,\
num_outdated=6,plot=False,_depth=7,_subsample=0.6,lrate=0.05,niter=100,logged=False,diffed=False,diff_lag=1,\
return_data=True,train_ensemble=False):
    
    if onehot:
        print("No onehot to work here")
    
    """
    if train_speed=='fast':
        niter=100
        lr=0.05
    else:
        niter=1500
        lr=0.025
    """   
    for col in ['item_category_id','Broad_cat','city','supercategory_id','platform_id','seasonal']:
        all_data[col]=all_data[col].astype('int64')
    
    for last_block in range(st_mth,end_mth):
        #last_block = 33
        print(last_block)
        
        X_train = all_data.loc[all_data['date_block_num'] <  last_block].drop(['target','Pred_target'], axis=1)
        X_test =  all_data.loc[all_data['date_block_num'] ==  last_block].drop(['target','Pred_target'], axis=1)
        
        cat_indices=np.where(X_train.columns.isin(['item_category_id','Broad_cat','city','supercategory_id','platform_id','seasonal']))[0]
        
        y_train = all_data.loc[all_data['date_block_num'] <  last_block, 'target'].clip(0,20).values
        y_test =  all_data.loc[all_data['date_block_num'] ==  last_block, 'target'].clip(0,20).values
        
        if logged:
            y_train2 = y_train.copy()
            y_test2 = y_test.copy()
            y_train = np.log(y_train+1)
            y_test =  np.log(y_test+1)
        elif diffed:
            y_train2 = y_train.copy()
            y_test2 = y_test.copy()
            y_train = all_data.loc[all_data['date_block_num'] <  last_block, 'target'].clip(0,20).values-all_data.loc[all_data['date_block_num'] <  last_block, 'target_lag_'+str(diff_lag)].clip(0,20).values
            y_test =  all_data.loc[all_data['date_block_num'] == last_block, 'target'].clip(0,20).values-all_data.loc[all_data['date_block_num'] == last_block, 'target_lag_'+str(diff_lag)].clip(0,20).values
        
        model=CatBoostRegressor(iterations=niter, depth=_depth, learning_rate=lrate, loss_function='RMSE',\
        has_time = True,subsample=_subsample)
        model.fit(X_train, y_train,cat_features=cat_indices.tolist(),plot=True)
        
        """
        Randomized HyperParameter Search
        
        cbc = CatBoostRegressor()
        param_dist = {"learning_rate": np.linspace(0,0.2,5),"max_depth": randint(6, 8)}
        rscv = RandomizedSearchCV(cbc , param_dist, scoring='neg_root_mean_squared_error', cv =5)
        rscv.fit(X_train,y_train)
        """
        
        pred_lgb = model.predict(X_test)
        pred_train=model.predict(X_train)
        
        if logged:
            pred_lgb=np.exp(pred_lgb)-1
            pred_train=np.exp(pred_train)-1
        elif diffed:
            pred_lgb=pred_lgb+all_data.loc[all_data['date_block_num'] ==  last_block, 'target_lag_'+str(diff_lag)].clip(0,20).values
            pred_train=pred_train+all_data.loc[all_data['date_block_num'] <  last_block, 'target_lag_'+str(diff_lag)].clip(0,20).values
        
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
        
        if logged or diffed:
            print('R-squared for TRAIN - %f' % r2_score(y_train2, pred_train))
            print('RMSE for TRAIN - %f' % mse(y_train2, pred_train,squared=False))
            print('R-squared for CV - %f' % r2_score(y_test2, pred_lgb))
            print('RMSE for CV - %f' % mse(y_test2, pred_lgb,squared=False))
        
        else:
            print('R-squared for TRAIN - %f' % r2_score(y_train, pred_train))
            print('RMSE for TRAIN - %f' % mse(y_train, pred_train,squared=False))
            print('R-squared for CV - %f' % r2_score(y_test, pred_lgb))
            print('RMSE for CV - %f' % mse(y_test, pred_lgb,squared=False))
        
        #all_data.loc[all_data['date_block_num'] < last_block,'Pred_target']=pred_train
        all_data.loc[all_data['date_block_num'] == last_block,'Pred_target']=pred_lgb.values
        
        del y_train,y_test
        if plot:
            plot_feature_importance(model.get_feature_importance(),all_data.loc[all_data['date_block_num'] ==  last_block].drop(['target','Pred_target'], axis=1).columns,'Catboost')
    if return_data:
        if not train_ensemble:
            return all_data.loc[(all_data['date_block_num']>=st_mth)&(all_data['date_block_num']<end_mth)]
        else:
            return all_data

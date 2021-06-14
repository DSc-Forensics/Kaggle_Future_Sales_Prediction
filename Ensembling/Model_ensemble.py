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

all_data['name_group_new_proportion_month'].fillna(0,inplace=True)

num_outdated=6

mods=['Cat','LGB','XG']

params=[{'_depth':4,'_subsample':0.325,'niter':200},{'_depth':3,'_subsample':0.68,'niter':140},{'_depth':7,'_subsample':0.28,'niter':250},\
{'_depth':5,'lrate':0.035,'niter':400},{'_depth':6,'_subsample':0.75,'lrate':0.04,'niter':150},{'_depth':5,'_subsample':0.5,'lrate':0.05,'niter':100}]
index_cols=['date_block_num','city','item_category_id','seasonal','target','Pred_target','target_lag_1']   
    
params=[{'_depth': 4, '_subsample': 0.35, 'niter': 250}, {'_depth': 8, '_subsample': 0.6, 'niter': 150}, {'_depth': 7, '_subsample': 0.3, 'niter': 250}, {'_depth': 5, 'lrate': 0.035, 'niter': 350}, {'_depth': 6, '_subsample': 0.75, 'lrate': 0.04, 'niter': 150}, {'_depth': 5, '_subsample': 0.5, 'lrate': 0.05, 'niter': 100}]
param={'_depth':8,'_subsample':0.75,'niter':1000,'lrate':0.004}

mods_to_run=[run_lgbm]
 
for param in params:
    count=0 
    
    for model in mods_to_run:   
        #out=pd.read_pickle('Train_ensemble_out/newfts_'+mods[count]+'_'+str(param['_depth'])+'_dep_'+str(param['niter'])+'.pkl')
        #out=model(all_data.copy(),sales_by_item_id,item_infos,st_mth=21,end_mth=35,diffed=True,diff_lag=12,**param)
        out=model(all_data.copy(),sales_by_item_id,item_infos,st_mth=21,end_mth=35,logged=False,diffed=False,diff_lag=1,\
        train_ensemble=False,**param)
        out[index_cols].to_pickle('Train_ensemble_out/Slow_newfts_'+mods[count]+'_'+str(param['_depth'])+'_dep_'+str(param['niter'])+'.pkl')
        count+=1
        del out
        gc.collect()
    param['niter']=100
    #out=run_xg(all_data.copy(),sales_by_item_id,item_infos,st_mth=21,end_mth=34,**param)
    #index_cols.remove('city')
    #out[index_cols].to_pickle('Train_ensemble_out/newfts_XG_'+str(param['_depth'])+'_dep_'+str(param['niter'])+'.pkl')
    count+=1
    #del out
    gc.collect()
    
#all_data=pd.read_parquet('Inputs/Train_ready_v2.gzip')
#item_infos=pd.read_parquet('Inputs/Item_infos.gzip')

del all_data
gc.collect()

train_cutoff=33

Xtrain=[]
Xtest=[]
x_cols=[]

for file in os.listdir("Train_ensemble_out"):

    if file.startswith('Train'):
        continue
    else:
        df=pd.read_pickle("Train_ensemble_out/"+file)
        
    xtr=df.loc[df['date_block_num']<=train_cutoff,'Pred_target'].clip(0,20)
    xtt=df.loc[(df['date_block_num']>train_cutoff)&(df['date_block_num']<35),'Pred_target'].clip(0,20)
    #xtt=df.loc[(df['date_block_num']==34),'Pred_target'].clip(0,20)
    
    x_cols.append('Pred_target_'+file.split('.')[0])
    
    Xtrain.append(np.array(xtr,dtype='int32'))
    Xtest.append(np.array(xtt,dtype='int32'))
    
Xtrain=pd.DataFrame(np.vstack(Xtrain).T,dtype=np.int32,columns=x_cols)
Xtest=pd.DataFrame(np.vstack(Xtest).T,dtype=np.int32,columns=x_cols)

ytr=df.loc[df['date_block_num']<=train_cutoff,'target'].clip(0,20)
ytt=df.loc[(df['date_block_num']>train_cutoff)&(df['date_block_num']<35),'target'].clip(0,20)
#ytt=df.loc[df['date_block_num']==34,'target'].clip(0,20)

del df
gc.collect()

model = LinearRegression(n_jobs=-1).fit(Xtrain, ytr)
#r_sq = model.score(Xtrain, ytr)
#print("R-Squared Train LR : "+str(r_sq))

pred_test = model.predict(Xtest)
pred_train=model.predict(Xtrain)

pred_test=pred_test.clip(0,20)
pred_train=pred_train.clip(0,20)

print('R-squared for TRAIN - %f' % r2_score(ytr, pred_train))
print('RMSE for TRAIN - %f' % mse(ytr, pred_train,squared=False))

print('R-squared for TEST - %f' % r2_score(ytt, pred_test))
print('RMSE for TEST - %f' % mse(ytt, pred_test,squared=False))

pred_test.mean()

print("-------------------------------------------------------")

best_tr_r2=0
best_tt_r2=0
best_tr_mse=2
best_tt_mse=2
for i in range(len(Xtrain.columns)):
    if r2_score(ytr, Xtrain[Xtrain.columns[i]])>best_tr_r2:
        best_tr_r2=r2_score(ytr, Xtrain[Xtrain.columns[i]])
        best_tr_r2_model=Xtrain.columns[i]
    if mse(ytr, Xtrain[Xtrain.columns[i]],squared=False)<best_tr_mse:
        best_tr_mse=mse(ytr, Xtrain[Xtrain.columns[i]],squared=False)
        best_tr_mse_model=Xtrain.columns[i]
        
for i in range(len(Xtest.columns)):
    if r2_score(ytt, Xtest[Xtest.columns[i]])>best_tt_r2:
        best_tt_r2=r2_score(ytt, Xtest[Xtest.columns[i]])
        best_tt_r2_model=Xtest.columns[i]
    if mse(ytt, Xtest[Xtest.columns[i]],squared=False)<best_tt_mse:
        best_tt_mse=mse(ytt, Xtest[Xtest.columns[i]],squared=False)

lgb_params = {'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':8, 
               'min_data_in_leaf': 20, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.04, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**9,
               'bagging_freq':1,
               'verbose':0,
               'num_iterations':100}

model = lgb.train(lgb_params, lgb.Dataset(Xtrain, label=ytr), 100)

pred_test = model.predict(Xtest)
pred_train=model.predict(Xtrain)

pred_test=pred_test.clip(0,20)
pred_train=pred_train.clip(0,20)

print('R-squared for TRAIN - %f' % r2_score(ytr, pred_train))
print('RMSE for TRAIN - %f' % mse(ytr, pred_train,squared=False))

print('R-squared for TEST - %f' % r2_score(ytt, pred_test))
print('RMSE for TEST - %f' % mse(ytt, pred_test,squared=False))

print("-------------------------------------------------------")

lgb_params = {'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':8, 
               'min_data_in_leaf': 20, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.004, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**9,
               'bagging_freq':1,
               'verbose':0,
               'num_iterations':1000,
               'categorical_feature':['name:item_category_id','name:seasonal','name:city']}

all_data=pd.read_pickle("Train_ensemble_out/newfts_Cat_4_dep_250.pkl")
newXtr=all_data.loc[all_data['date_block_num']<=train_cutoff].drop(['target','Pred_target'],axis=1)
newXtt=all_data.loc[(all_data['date_block_num']>train_cutoff)&(all_data['date_block_num']<35)].drop(['target','Pred_target'],axis=1)

Xtrain[newXtr.columns]=newXtr.values
Xtest[newXtt.columns]=newXtt.values

model = lgb.train(lgb_params, lgb.Dataset(Xtrain, label=ytr,categorical_feature=['item_category_id','seasonal','city']), 1000)

pred_test = model.predict(Xtest)
pred_train=model.predict(Xtrain)

pred_test=pred_test.clip(0,20)
pred_train=pred_train.clip(0,20)

print('R-squared for TRAIN - %f' % r2_score(ytr, pred_train))
print('RMSE for TRAIN - %f' % mse(ytr, pred_train,squared=False))

print('R-squared for TEST - %f' % r2_score(ytt, pred_test))
print('RMSE for TEST - %f' % mse(ytt, pred_test,squared=False))

out_test_items=pd.read_csv("Outdated_test_items.csv")
sub=pd.read_csv("Inputs/sample_submission.csv")
test=pd.read_csv("Inputs/test.csv")

sub['item_cnt_month']=pred_test
sub['item_cnt_month']=sub['item_cnt_month'].clip(0,20)

sub['item_cnt_month'].mean()

#simple_sub=pd.read_csv("Subs/slow_6rollmean_bomb_simple_sub_bigdata_clipped.csv")
sub=sub.merge(test,on='ID')

sub=pd.merge(sub,out_test_items,how='left',on=['ID','shop_id','item_id'])

sub2=sub.copy()
sub2['item_cnt_month']=np.where(sub2['open']==sub2['open'],0,sub2['item_cnt_month'])

sub2['item_cnt_month'].mean()

sub2.set_index('ID')['item_cnt_month'].to_csv("Subs/slow_ensem_tfidf_rollmeaned_nobc_simple_sub_smalldata_clipped.csv")


"""    
model = LinearRegression(n_jobs=-1).fit(x, y)
r_sq = model.score(x, y)
y_pred = model.predict(x)
"""
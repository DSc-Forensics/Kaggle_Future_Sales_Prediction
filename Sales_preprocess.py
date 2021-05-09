# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:12:51 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales")

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

sales=pd.read_hdf("Inputs/Translated_sales.hdf")
test=pd.read_csv("Inputs/test.csv")

#Change of redundant shops
sales.loc[sales['shop_id']==0,'Trans_shops']='Yakutsk Ordzhonikidze, 56 '
sales.loc[sales['Trans_shops']=='Yakutsk Ordzhonikidze, 56 ','shop_id']=57
sales.loc[sales['shop_id']==1,'Trans_shops']='Yakutsk shopping center "Central" '
sales.loc[sales['Trans_shops']=='Yakutsk shopping center "Central" ','shop_id']=58
sales.loc[sales['shop_id']==11,'Trans_shops']='Zhukovsky st. Chkalova 39m? '
sales.loc[sales['Trans_shops']=='Zhukovsky st. Chkalova 39m? ','shop_id']=10
sales.loc[sales['shop_id']==40,'Trans_shops']='Rostnone TRK "Megacentr Horizont"'
sales.loc[sales['Trans_shops']=='Rostnone TRK "Megacentr Horizont"','shop_id']=39

del sales['shop_name']

sales.loc[sales['date_block_num']==34,'item_price']=0.01
sales = sales[(sales['item_price'] > 0) & (sales['item_price'] < 51000)]
sales.loc[sales['date_block_num']==34,'item_cnt_day']=0
sales = sales[sales['item_cnt_day'] <= 1000]

sales=sales[(sales['shop_id']!=9)&(sales['shop_id']!=20)]

sales['city']=sales['Trans_shops'].str.split(' ').str[0]
sales.loc[sales['city']=='Voronež','city']='Voronezh'
sales.loc[sales['city']=='SPb','city']='SPB'
sales['city']=np.where(sales['city'].str.contains('online'),'Internet',sales['city'])
sales['city']=np.where(sales['city'].str.contains('Online'),'Internet',sales['city'])
sales['city']=np.where(sales['city']=='Rostnone','Rostovnadon',sales['city'])

city_info = pd.read_csv('Inputs/city_info.csv')
city_info['Trans_city']=city_info['Trans_city'].str.replace(' ','')
city_info=city_info.loc[city_info.Trans_city.isin(sales['city'].unique())]
city_info['city_size'].fillna(city_info['city_size'].mean(),inplace=True)
city_sizes=city_info.set_index('Trans_city')['city_size'].to_dict()
sales['city_size']=sales['city'].map(city_sizes)

cal=pd.read_csv("Inputs/calendar.csv")
cal['date_block_num']=(cal['year']-2013)*12+cal['month']-1
cal['hdays']=cal['mdays']-cal['wdays']
sales=pd.merge(sales,cal[['date_block_num','mdays','hdays']],how='left',on='date_block_num')

item_block_cnt = sales[sales['date_block_num']<34].groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum()
items_to_drop = item_block_cnt[item_block_cnt <= 0].index
sales = sales[~sales.set_index(['date_block_num', 'shop_id', 'item_id']).index.isin(items_to_drop)]

sales_by_item_id = sales[sales['date_block_num']<34].pivot_table(index=['item_id'],values=['item_cnt_day'], 
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'

outdated_items = sales_by_item_id[sales_by_item_id.loc[:,'27':].sum(axis=1)==0]
print('Outdated items:', len(outdated_items))

print('Outdated items in test set:', len(test[test['item_id'].isin(outdated_items['item_id'])]))

outdated_test_items=test[test['item_id'].isin(outdated_items['item_id'])]
outdated_test_items['open']=0
outdated_test_items.set_index('ID').to_csv("Outdated_test_items.csv")

sales.drop(['item_category_name','Trans_cats', 'Trans_shops','ID','item_name','Trans_item'],axis=1,inplace=True)

sales = downcast_dtypes(sales)
sales.to_hdf("Inputs/Translated_sales_preprocessed.hdf",key='df')

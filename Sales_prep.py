
"""
Created on Sun May  9 13:57:44 2021

@author: Spectre
"""

import pandas as pd
import os

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales")

from google_trans_new import google_translator  
translator = google_translator()

sales=pd.read_csv("Inputs/sales_train.csv")     
item_cat=pd.read_csv("Inputs/item_categories.csv")
item=pd.read_csv("Inputs/items.csv")
sub=pd.read_csv("Inputs/sample_submission.csv")
shops=pd.read_csv("Inputs/shops.csv")
test=pd.read_csv("Inputs/test.csv")

sales=sales.join(item.set_index('item_id'),on='item_id',how='left')
sales=sales.join(item_cat.set_index('item_category_id'),on='item_category_id',how='left')
sales=sales.join(shops.set_index('shop_id'),on='shop_id',how='left')
sales['Revenue']=sales['item_price']*sales['item_cnt_day']

test=test.join(item.set_index('item_id'),on='item_id',how='left')
test=test.join(item_cat.set_index('item_category_id'),on='item_category_id',how='left')
test=test.join(shops.set_index('shop_id'),on='shop_id',how='left')
test['date_block_num']=34
sales=sales.append(test)

sales=sales.loc[sales['item_category_id'].isin(test['item_category_id'])]
sales.to_hdf("Inputs/Total_shop_cat_translated.hdf",key='df')

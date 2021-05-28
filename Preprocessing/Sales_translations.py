# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:51:29 2021

@author: PC
"""

import pandas as pd
import os
import time
import numpy as np

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales")

from google_trans_new import google_translator  
translator = google_translator()

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
    
item_cat=pd.read_csv("Inputs/item_categories.csv")
item=pd.read_csv("Inputs/items.csv")
sub=pd.read_csv("Inputs/sample_submission.csv")
shops=pd.read_csv("Inputs/shops.csv")
test=pd.read_csv("Inputs/test.csv")

item_trans=pd.read_csv("Inputs/Completed_item_translations.csv")
item_trans.columns=['item_name','Trans_item']
sales=pd.read_hdf("Inputs/sales.hdf") 

cat_translations = {}
unique_elements = item_cat['item_category_name'].unique()
for element in unique_elements:
    # add translation to the dictionary
    cat_translations[element] = translator.translate(element)
    
item_cat['Trans_cat']=item_cat['item_category_name'].map(cat_translations)
sales['Trans_cats']=sales['item_category_name'].map(cat_translations)

shop_translations = {}
unique_elements = shops['shop_name'].unique()
for element in unique_elements:
    # add translation to the dictionary
    shop_translations[element] = translator.translate(element)

shops['Trans_shops']=shops['shop_name'].map(shop_translations)
sales['Trans_shops']=sales['shop_name'].map(shop_translations)
sales['Broad_cat']=sales['Trans_cats'].str.split('-').str[0]

shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()
shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
shop_types=shops.set_index('shop_id')['shop_type'].to_dict()
sales['shop_type']=sales['shop_id'].map(shop_types)

item_translations = {}
unique_elements = sales['item_name'].unique()
for element in unique_elements:
    # add translation to the dictionary
    # Need to add a sleep comment, because too many calls to google translator in a short period of time leads to being blocked out
    time.sleep(1.25)
    if element not in item_translations:
       item_translations[element] = translator.translate(element)

#It takes a lot of time to save item translations. Also very voluminous data, good idea to save it for future analysis
pd.DataFrame.from_dict(item_translations,orient='index').to_csv("Completed_item_translations.csv")
item['Trans_item']=item['item_name'].map(item_translations)
item['Broad_item']=item['Trans_item'].str.split(' ').str[0:2].str.join(',').str.replace(',',' ')

#Broad item seems more interesting rather than translated item, so merging that column on sales for now
sales=pd.merge(sales,item[['item_name','Broad_item']],how='left',on='item_name')

sales = downcast_dtypes(sales)
sales.to_hdf("Inputs/Translated_sales.hdf",key='df')


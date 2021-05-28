# Kaggle_Future_Sales_Prediction
Predicting Future Sales for a Russian software retail company 1C (https://www.kaggle.com/c/competitive-data-science-predict-future-sales). Part of the final project for coursera course "How to Win a Data Science Competition"(https://www.coursera.org/learn/competitive-data-science/home)

Given a timeseries starting 2013-2015 Oct (**3 million rows of data**) for sales of 1C on various items in it's multiple shop outlets, the ask is to predict monthly sales for the various shop-item pairs in Nov 2015.
This submssion was good for a **0.837 RMSE score and 36/11484 place (top 0.3%)** according to the leaderboard as on 05/28/2021. It was cross validated using roll forward validation (not included in this repo) and forecasted using the Decision Trees LGBM model. **Work on this competition is still ongoing**, so expect updates in the coming times..

Please note the sequence to go through the code in the folders is as follows-;

1. Preprocessing
2. Monthly Aggregation (performed in 2 ways and thus repo contains 2 folders- one for each method)
3. Ensembling (Still Work In Progress)
4. Test Run

Or you can peek at the complete pipeline in 1 notebook-> "Complete_Pred_Pipeline.ipynb".

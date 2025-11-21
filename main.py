import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from tqdm import tqdm
from preprocessing import preprocessing, build_training_data
from pairs import find_comovement_pairs
from train_predict import train, predict 
from xgboost import XGBRegressor


data_path = './train.csv'
pivot = preprocessing(data_path)
pairs = find_comovement_pairs(pivot)

print("탐색된 공행성쌍 수:", len(pairs))
df_train_model = build_training_data(pivot, pairs)


trained_model =train(df_train_model)
df_pred = predict(pivot, pairs, trained_model)
df_pred.to_csv('./submit.csv', index=False)
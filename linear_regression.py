import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

white_wine_df = pd.read_csv('winequality-white.csv', delimiter=';')

outlier = white_wine_df.loc[white_wine_df['free sulfur dioxide'] > 250]

# print(outlier)

white_wine_df = white_wine_df.drop(outlier.index)
# print(white_wine_df[white_wine_df.isna().any(axis=1)])
total_SO2_df = white_wine_df['total sulfur dioxide'].values.reshape(-1, 1)
free_SO2_df = white_wine_df['free sulfur dioxide'].values.reshape(-1, 1)

total_SO2_train, total_SO2_test, free_S02_train, free_SO2_test = train_test_split(total_SO2_df, free_SO2_df, test_size=0.2)

model = LinearRegression()  # create model
model.fit(total_SO2_train, free_S02_train)  # train to fit the data points

print(model.score(total_SO2_test, free_SO2_test))


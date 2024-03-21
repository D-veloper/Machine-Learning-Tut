import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

white_wine_df = pd.read_csv('winequality-white.csv', delimiter=';')

outlier = white_wine_df.loc[white_wine_df['free sulfur dioxide'] > 250]

# print(outlier)

white_wine_df = white_wine_df.drop(outlier.index)

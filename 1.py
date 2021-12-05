import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



hm_data = pd.read_csv("train.csv")
hm_data = hm_data.fillna(-1)
y = hm_data.SalePrice

fetures = ['MSSubClass',
            'LotArea',
            'OverallQual',
            'OverallCond',
            'YearBuilt',
            'YearRemodAdd',
            '1stFlrSF',
            '2ndFlrSF',
            'LowQualFinSF',
            'GrLivArea',
            'FullBath',
            'HalfBath',
            'BedroomAbvGr',
            'KitchenAbvGr',
            'TotRmsAbvGrd',
            'Fireplaces',
            'WoodDeckSF',
            'OpenPorchSF',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
            'PoolArea',
            'MiscVal',
            'MoSold',
            'YrSold']

X = hm_data[fetures]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# creating a model using DecisionTreeRegressor
# random_state is provided for model reproducability
rf_model = RandomForestRegressor(random_state=1, n_estimators=700)
gb_model = GradientBoostingRegressor(random_state=1, n_estimators=700)

# fitting model
rf_model.fit(train_X,train_y)

# making predictions
rf_predictions = rf_model.predict(val_X)
# print(predictions)
gb_model.fit(train_X,train_y)
gb_predictions = gb_model.predict(val_X)
predictions = (rf_predictions + gb_predictions)/2


# Result review
# print(val_y.head(), rf_predictions[:5])
print("RMSE validation for RandomForestRegressor:{:,.0f}".format(np.sqrt(mean_squared_error(val_y, rf_predictions))))
print("RMSE validation for GradientBoostingRegressor:{:,.0f}".format(np.sqrt(mean_squared_error(val_y, gb_predictions))))
print("RMSE validation for avg of RFM & GBM:{:,.0f}".format(np.sqrt(mean_squared_error(val_y, predictions))))

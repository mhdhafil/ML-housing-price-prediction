import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



hm_data = pd.read_csv("train.csv")
y = hm_data.SalePrice

fetures = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = hm_data[fetures]
# print(X.describe())
# print(X.head())
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# creating a model using DecisionTreeRegressor
# random_state is provided for model reproducability
model = DecisionTreeRegressor()

# fitting model
model.fit(train_X,train_y)

# making predictions
predictions = model.predict(val_X)
# print(predictions)

# Result review
print(val_y.head(), predictions[:5])

print(mean_absolute_error(val_y, predictions))

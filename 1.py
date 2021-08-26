import pandas as pd
from sklearn.tree import DecisionTreeRegressor



hm_data = pd.read_csv("train.csv")
y = hm_data.SalePrice

fetures = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = hm_data[fetures]
# print(X.describe())

# print(X.head())


# creating a model using DecisionTreeRegressor
# random_state is provided for model reproducability
model = DecisionTreeRegressor(random_state=1)

# fitting model
model.fit(X,y)

# making predictions
predictions = model.predict(X)
# print(predictions)

# Result review
print(y.head(), predictions[:5])


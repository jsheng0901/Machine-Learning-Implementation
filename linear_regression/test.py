import datetime

from sklearn.metrics import mean_squared_error
from linear_regression import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

# loading regression data
data = datasets.load_diabetes()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build mode
start = datetime.datetime.now()
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = datetime.datetime.now()
# get metric
mse = mean_squared_error(y_test, y_pred)

print(f"Running model took {end - start} seconds")
print(f"Training date size is [{len(y)}, {len(X[0])}]")
print(f"MSE is {mse}")

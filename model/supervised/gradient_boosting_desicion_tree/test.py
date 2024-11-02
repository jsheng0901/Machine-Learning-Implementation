import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from model.supervised.gradient_boosting_desicion_tree import GradientBoostingBinaryClassifier, GradientBoostingRegressor

# loading classification data
data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("-- Classification Tree --")
# build mode
start = datetime.datetime.now()
model = GradientBoostingBinaryClassifier(n_estimators=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = datetime.datetime.now()
# get metric
accuracy = accuracy_score(y_test, y_pred)

print(f"Running model took {end - start} seconds")
print(f"Training date size is [{len(y)}, {len(X[0])}]")
print(f"Accuracy is {accuracy}")

# loading regression data
data = datasets.load_diabetes()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("-- Regression Tree --")
# build mode
start = datetime.datetime.now()
model = GradientBoostingRegressor(n_estimators=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = datetime.datetime.now()
# get metric
mse = mean_squared_error(y_test, y_pred)

print(f"Running model took {end - start} seconds")
print(f"Training date size is [{len(y)}, {len(X[0])}]")
print(f"MSE is {mse}")

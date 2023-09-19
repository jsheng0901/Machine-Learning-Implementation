import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from decision_tree import RegressionTree, ClassificationCARTTree, ClassificationID3Tree

# loading classification data
data = datasets.load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("-- Classification CART Tree --")
# build mode
start = datetime.datetime.now()
model = ClassificationCARTTree()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = datetime.datetime.now()
# get metric
accuracy = accuracy_score(y_test, y_pred)

print(f"Running model took {end - start} seconds")
print(f"Training date size is [{len(y)}, {len(X[0])}]")
print(f"Accuracy is {accuracy}")

print("-- Classification ID3 Tree --")
# build mode
start = datetime.datetime.now()
model = ClassificationID3Tree()
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
model = RegressionTree()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = datetime.datetime.now()
# get metric
mse = mean_squared_error(y_test, y_pred)

print(f"Running model took {end - start} seconds")
print(f"Training date size is [{len(y)}, {len(X[0])}]")
print(f"MSE is {mse}")

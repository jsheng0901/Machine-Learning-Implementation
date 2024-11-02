import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model.supervised.logistic_regression import LogisticRegression
from utils.regularization import Ridge

# loading data
data = datasets.load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build mode
start = datetime.datetime.now()
model = LogisticRegression(regularization=Ridge)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = datetime.datetime.now()
# get metric
accuracy = accuracy_score(y_test, y_pred)

print(f"Running model took {end - start} seconds")
print(f"Training date size is [{len(y)}, {len(X[0])}]")
print(f"Accuracy is {accuracy}")

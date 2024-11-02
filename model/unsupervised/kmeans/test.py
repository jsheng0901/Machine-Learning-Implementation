import datetime
from sklearn.datasets import make_blobs
from model.unsupervised.kmeans import KMeans

# generate data
data, _ = make_blobs(100000, 10)

# build model
start = datetime.datetime.now()
kmeans = KMeans(k=5)
kmeans_output = kmeans.predict(data)
end = datetime.datetime.now()
print(f"Running model took {end - start} seconds")

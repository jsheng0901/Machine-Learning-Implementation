import datetime
from sklearn.datasets import make_blobs
from gaussian_mixture_model.gmm import GaussianMixtureModel

# generate data
data, _ = make_blobs(10000, 10)

# build model
start = datetime.datetime.now()
gmm_model = GaussianMixtureModel(k=5)
gmm_output = gmm_model.fit(data)
end = datetime.datetime.now()
print(f"Running model took {end - start} seconds")

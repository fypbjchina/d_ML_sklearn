# Nearest Neighbors Transformer
# d01_6_60_1_nearestneighbors_transformer.py

import tempfile
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
cache_path = tempfile.gettempdir()  # we use a temporary folder here

X, _ = make_regression(n_samples=50, n_features=25, random_state=0)
estimator = make_pipeline(
    KNeighborsTransformer(mode='distance'),
    Isomap(n_components=3, metric='precomputed'),
    memory=cache_path)

X_embedded = estimator.fit_transform(X)
X_embedded.shape
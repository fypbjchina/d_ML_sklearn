# Example of non-cached pipeline with PCA and SVC
# d06_1_10_non_cached_pipeline.py

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits

X_digits, y_digits = load_digits(return_X_y=True)
pca1 = PCA(n_components=10)
svm1 = SVC()

# Without caching, parameter memory=None
pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])

pipe_fit = pipe.fit(X_digits, y_digits)
print(pipe_fit)

# The pca instance can be inspected directly
pca1.components_.shape
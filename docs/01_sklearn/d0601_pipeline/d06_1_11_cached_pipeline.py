# Example of cached pipeline with PCA and SVC
# d06_1_11_cached_pipeline.py

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits

cachedir = mkdtemp()
pca2 = PCA(n_components=10)
svm2 = SVC()

# with caching, parameter memory=cachedir
cached_pipe = Pipeline([('reduce_dim', pca2), ('clf', svm2)],
                       memory=cachedir)
cached_pipe_fit = cached_pipe.fit(X_digits, y_digits)
print(cached_pipe_fit)

# The pca instance cannot be inspected directly
# pca2.components_.shape

# .named_steps to access the fitted transformers of the cached pipeline
cached_pipe.named_steps['reduce_dim'].components_.shape

# Remove the cache directory
# rmtree(cachedir)
# Example of pipeline with PCA and SVC
# d06_1_9_pipeline_memory.py

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

estimators = [('reduce_dim', PCA()), ('clf', SVC())]
cachedir = mkdtemp()
pipe = Pipeline(estimators, memory=cachedir)

print(pipe)
print(cachedir)

# Clear the cache directory when you don't need it anymore
print(rmtree(cachedir))
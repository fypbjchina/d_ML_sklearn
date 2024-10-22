# access to nested parameters
# d06_1_6_nested_parameters_access.py

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

pipe = Pipeline(steps=[("reduce_dim", PCA()), ("clf", SVC())])

# parameter C of the clf estimator
print(pipe.set_params(clf__C=10))
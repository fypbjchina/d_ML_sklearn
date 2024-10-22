# access to nested parameters
# d06_1_8_nested_parameters_passthrough.py

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

pipe = Pipeline(steps=[("reduce_dim", PCA()), ("clf", SVC())])

param_grid = dict(reduce_dim=['passthrough', PCA(5), PCA(10)],
                  clf=[SVC(), LogisticRegression()],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
print(grid_search)
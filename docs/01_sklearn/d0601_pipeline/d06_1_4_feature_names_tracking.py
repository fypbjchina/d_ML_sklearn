# track feature names
# d06_1_4_feature_names_tracking.py

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

iris = load_iris()
pipe = Pipeline(steps=[
    ('reduce_dim', PCA(n_components=2)),
    ('select', SelectKBest(k=2)),
    ('clf', LogisticRegression())])
print(pipe)

pipe.fit(iris.data, iris.target)

# feature names for steps with get_feature_names_out
for name, step in pipe.steps:
    if hasattr(step, 'get_feature_names_out'):
        print(f"Step name --> object + feature_names:\t {name} --> {step} + {step.get_feature_names_out()}")
    else:
        print(f"Step name --> object:\t {name} --> {step}")

a = pipe[0].get_feature_names_out()
b = pipe[1].get_feature_names_out()
print(a)
print(b)
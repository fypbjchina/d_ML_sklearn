# track feature names
# d06_1_5_feature_names_customize.py

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

iris = load_iris()
pipe = Pipeline(steps=[
    ('select', SelectKBest(k=2)),
    ('clf', LogisticRegression())])
print(pipe)

pipe.fit(iris.data, iris.target)

# customize feature names
pipe[:-1].get_feature_names_out(iris.feature_names)
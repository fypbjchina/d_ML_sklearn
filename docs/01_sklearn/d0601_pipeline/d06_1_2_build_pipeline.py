# build a pipeline
# d06_1_2_build_pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

estimators = [('reduce_dim', PCA()), ('clf', SVC()), ('feature_selection', SelectKBest())]

pipe = Pipeline(estimators)
pipe
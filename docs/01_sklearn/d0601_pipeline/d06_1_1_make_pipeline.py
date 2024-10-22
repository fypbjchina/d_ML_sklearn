# make pipeline
# d06_1_1_make_pipeline.py

from sklearn.pipeline import make_pipeline

make_pipeline(PCA(), SVC(), SelectKBest())
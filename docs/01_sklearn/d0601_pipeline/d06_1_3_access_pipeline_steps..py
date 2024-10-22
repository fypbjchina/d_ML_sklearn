# access pipeline steps
# d06_1_3_access_pipeline_steps.py

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

estimators = [('reduce_dim', PCA()), ('clf', SVC()), ('feature_selection', SelectKBest())]
pipe = Pipeline(estimators)

# []: access the steps by slicing
a = pipe[:2]
b = pipe[-2:]

print(a)
print(b)

# [idx]: access the step by index
c = pipe.steps[0]
d = pipe[0]
e = pipe['clf']

print(c)
print(d)
print(e)

# access pipeline steps
steps_slice = pipe.steps[0:3]

for name, step in steps_slice:
    print(f"Step name --> object:\t {name} --> {step}")
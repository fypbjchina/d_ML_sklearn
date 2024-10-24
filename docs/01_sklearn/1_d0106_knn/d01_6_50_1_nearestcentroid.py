# Nearest centroid classifier
# d01_6_50_1_nearestcentroid.py

from sklearn.neighbors import NearestCentroid
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
print(clf)

clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))
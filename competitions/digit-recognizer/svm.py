

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
labeled_images = pd.read_csv('./input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# test_images[test_images>0]=1
# train_images[train_images>0]=1
test_images = test_images / 255.0
train_images = train_images / 255.0
# img=train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap='binary')
# plt.title(train_labels.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
a = clf.score(test_images,test_labels)
print(a)
test_data=pd.read_csv('./input/test.csv')
# test_data[test_data>0]=1
test_data = test_data/255.0
results=clf.predict(test_data)

print(results)
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)




C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
X = images
y = labels
grid.fit(X, y)

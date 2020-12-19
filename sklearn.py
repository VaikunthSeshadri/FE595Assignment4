import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings("ignore")

boston = load_boston()

print(boston.DESCR)

bosdf = pd.DataFrame(boston.data, columns = boston.feature_names)

bosdf.head()

plt.subplots(figsize=(20,15))
sns.heatmap(bosdf.corr(), annot=True)

bosdf.drop(['INDUS', 'NOX', 'TAX', 'AGE'], axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts

X = bosdf.drop(['LSTAT'],1)
y = bosdf['LSTAT']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3,random_state=42)

lr = LinearRegression(copy_X= True, fit_intercept = True)

lr.fit(X_train, y_train)

lr_pred= lr.predict(X_test)

lr.score(X_test,y_test)

importance = lr.coef_

for i,j in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,j))

feature_names = bosdf.columns

plt.bar([x for x in range(len(importance))], importance)
plt.xticks(range(bos.shape[1]), feature_names)
plt.xticks(rotation=90)
plt.xlim([-1, bosdf.shape[1]])
plt.show()



warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris

ld = load_iris()
irisdf = pd.DataFrame(ld.data, columns=ld.feature_names)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# k means determine k
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(irisdf)
    kmeanModel.fit(irisdf)
    distortions.append(sum(np.min(cdist(irisdf, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / irisdf.shape[0])

    # Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Repeating same steps for wine dataset

from sklearn.datasets import load_wine

ld2 = load_wine()
wine = pd.DataFrame(ld2.data, columns=ld2.feature_names)

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(wine)
    kmeanModel.fit(wine)
    distortions.append(sum(np.min(cdist(wine, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / wine.shape[0])

    plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

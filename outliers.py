import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.font_manager
import numpy as np 
from scipy import stats
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

dataset=pd.read_excel('player_playoff_career_avg.xlsx')
X=dataset.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values


#Spiltting data into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test=train_test_split(X, test_size=0.2, random_state=0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
#We use fit and transform only for training set. Do not need to user for test set
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


random_state = np.random.RandomState(42)
outliers_fraction = 0.010

classifiers = {
        
       'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
       # 'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        #'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False,        random_state=random_state),
       # 'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
       # 'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
         # 'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}


for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1
        
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))
    threshold = stats.scoreatpercentile(scores_pred,100 *       outliers_fraction)
    # copy of dataframe
    dfx = dataset
    dfx['outlier'] = y_pred.tolist()
    dfy=dfx[ dfx['outlier']==1]       
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data.csv')
df.head(7)


df.shape


df.isna().sum()
df.isnull().sum()




df = df.dropna(axis=1)


df.shape


df['diagnosis'].value_counts()


sns.countplot(df['diagnosis'], label="Count")
plt.show()




from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()
df.iloc[:, 1] = labelencoder_Y.fit_transform(df.iloc[:, 1].values)
print(labelencoder_Y.fit_transform(df.iloc[:, 1].values))
df = df.set_index('id')



sns.pairplot(df, hue="diagnosis")



df.head(5)


df.corr()
print(df.corr())



plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt='.0%')
plt.show()

df.shape
print(df.shape)

Y = df['diagnosis'].values
X =  df.drop('diagnosis', axis=1).values


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)


from sklearn.feature_selection import RFECV


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=123)


print(Y)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




def models(X_train, Y_train):

    from sklearn.linear_model import LogisticRegression
    # log = LogisticRegression(random_state=10)
    log = LogisticRegression(solver='lbfgs', penalty='l2', random_state=123)
    log.fit(X_train, Y_train)



    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
    knn.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_lin = SVC(kernel='rbf', random_state=123, C=1.0,tol=0.001,class_weight='balanced')
    svc_lin.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='linear', random_state=123,C=1,tol=0.1)
    svc_rbf.fit(X_train, Y_train)

    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)


    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=10, splitter='random',min_samples_leaf=2,max_depth=30,min_samples_split=5,max_features='auto')
    tree.fit(X_train, Y_train)


    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(bootstrap=True, criterion='gini', random_state=123,
                                    max_depth=40,min_samples_leaf=2,max_features='auto',min_samples_split=10, n_estimators=2000)

    forest.fit(X_train, Y_train)


    # print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


model = models(X_train, Y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test))

    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    print(cm)
    print('Model[{}] Testing Accuracy = "{}!"'.format(i, (TP + TN) / (TP + TN + FN + FP)))
    print()
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
    print('Model ', i)

    print(classification_report(Y_test, model[i].predict(X_test)))

    print(accuracy_score(Y_test, model[i].predict(X_test)))
    print()


# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state = 42)
# from pprint import pprint
# # Look at parameters used by our current forest
# print('Parameters currently in use:\n')
# pprint(rf.get_params())
# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# pprint(random_grid)
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train, Y_train)


from sklearn.svm import SVC
import pickle
from sklearn import datasets

from sklearn.externals import joblib
classifier = SVC()

joblib.dump(model, "model100.pkl")





classifier = joblib.load("model100.pkl")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.externals import joblib


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=569, centers=30, n_features=30, random_state=100)


model = SVC(kernel='rbf', random_state=123, C=6,tol=0.1,class_weight='balanced')


model.fit(X, Y)


pickle.dump(model, open('model100.pkl','wb'))




model = pickle.load(open('model100.pkl','rb'))
print()
print(model.predict([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))


















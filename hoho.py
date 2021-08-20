

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
# Import all needed libraries and sublibraries
from keras.utils import to_categorical






df = pd.read_csv('METABRIC_RNA_Mutation.csv')
df.info()
print(df.info())


'''
df = df.astype({"pam50_claudin_low_subtype":'category',"type_of_breast_surgery":'category',"cancer_type":'category',"cancer_type_detailed":'category',"cellularity":'category',"er_status_measured_by_ihc":'category',"er_status":'category',"her2_status_measured_by_snp6":'category',"her2_status":'category',"tumor_other_histologic_subtype":'category',"inferred_menopausal_state":'category',"integrative_cluster":'category',"primary_tumor_laterality":'category',"oncotree_code":'category',"pr_status":'category',"3_gene_classifier_subtype":'category',"death_from_cancer":'category'})
'''







df.isna().sum()
df.isnull().sum()

def tran_pam50_claudin_low_subtype(x):
    if x == 'cloudinlow':
        return 0
    if x == 'LumA':
        return 1
    if x == 'LumB':
        return 2
    if x == 'Her2':
        return 3
    if x == 'Basal':
        return 4
    if x == 'Normal':
        return 5
df['tran_pam50_claudin_low_subtype'] = df['pam50_claudin_low_subtype'].apply(tran_pam50_claudin_low_subtype)

'''
df.to_csv("/Users/nurhaqimkamaruddin/PycharmProjects/untitled5/metanomics_filtered.csv",index=False)
df1 = pd.read_csv('/Users/nurhaqimkamaruddin/PycharmProjects/untitled5/metanomics_filtered.csv.csv')

print(df1.shape)
df1.head()
print(df1.head())

'''



df.shape
print(df.shape)
df.info()
print(df.info())



null_counts = df.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))
object_columns_df = df.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])
cols = ['pam50_claudin_low_subtype','er_status','her2_status_measured_by_snp6','her2_status','inferred_menopausal_state','pr_status','death_from_cancer']
for name in cols:
    print(name,':')
    print(object_columns_df[name].value_counts(),'\n')
drop_cols = ['oncotree_code']

df = df.drop(drop_cols,axis=1)



df.shape
print(df.shape)

mapping_dict = {
"pam50_claudin_low_subtype": {
"LumA": 1,
"LumB": 2,
"Her2": 3,
"Basal": 4,
"claudinlow": 5,
"Normal": 6,
"NC": 7,
},
"her2_status_measured_by_snp6":{
"NEUTRAL": 0,
"GAIN": 2,
"LOSS": 3,
"UNDEF": 4,
},
"death_from_cancer ":{
"Living": 0,
"Died of Disease": 1,
"Died of Other Causes": 2,
}
}
df = df.replace(mapping_dict)
df[['pam50_claudin_low_subtype','her2_status_measured_by_snp6']].head()
nominal_columns = ["er_status", "her2_status", "inferred_menopausal_state", "pr_status","death_from_cancer",]
dummy_df = pd.get_dummies(df[nominal_columns])
df = pd.concat([df, dummy_df], axis=1)
df = df.drop(nominal_columns, axis=1)
df = df.dropna()

df.shape
print(df.shape)
df.info()
print(df.info())

df['chemotherapy'].value_counts()


sns.countplot(df['chemotherapy'], label="Count")
plt.show()


'''
from sklearn.preprocessing import OneHotEncoder

onehotencoder_Y = OneHotEncoder()
df.iloc[:,4] = onehotencoder_Y.fit_transform(df.iloc[:,4].values)



sns.pairplot(df, hue="diagnosis")



df.head(5)


df.corr()
'''


df = df.rename(columns={'er_status':'ER_STATUS'})
'''
from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()
df.loc[:,4] = labelencoder_Y.fit_transform(df.loc[:,4].values)
print(labelencoder_Y.fit_transform(df.iloc[:,4].values))

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt='.0%')
plt.show()

'''

X = df.iloc[:, 1:22].values
Y = df.iloc[:, 0].values

'''

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# feature extraction
model = RandomForestClassifier(criterion='gini')
rfe = RFE(model, 23)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
'''

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=100)




from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



def models(X_train, Y_train):

    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=100)
    log.fit(X_train, Y_train)


    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=50, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state=100)
    svc_lin.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=100)
    svc_rbf.fit(X_train, Y_train)


    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)


    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=100)
    tree.fit(X_train, Y_train)


    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100)
    forest.fit(X_train, Y_train)

    # print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


model = models(X_train, Y_train)


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







# Print Prediction of Random Forest Classifier model


from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn import datasets

from sklearn.externals import joblib
classifier = KNeighborsClassifier()

joblib.dump(model, "model33.pkl")





classifier = joblib.load("model33.pkl")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1683, centers=10, n_features=23, random_state=100)

model = KNeighborsClassifier(metric='minkowski')



model.fit(X, Y)

pickle.dump(model, open('model33.pkl','wb'))





model = pickle.load(open('model33.pkl','rb'))
print()
print(model.predict([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23]]))


A = df.iloc[:, 0:21].values
B = df.iloc[:, 9].values

'''

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# feature extraction
model = RandomForestClassifier(criterion='gini')
rfe = RFE(model, 23)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
'''

from sklearn.model_selection import train_test_split

A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.30, random_state=100)




from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
A_train = sc.fit_transform(A_train)
A_test = sc.transform(A_test)



def models(A_train, B_train):

    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=100)
    log.fit(A_train, B_train)


    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=50, metric='minkowski', p=2)
    knn.fit(A_train, B_train)


    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state=100)
    svc_lin.fit(A_train, B_train)


    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=100)
    svc_rbf.fit(A_train, B_train)


    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(A_train, B_train)


    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=100)
    tree.fit(A_train, B_train)


    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100)
    forest.fit(A_train, B_train)

    # print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(A_train, B_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(A_train, B_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(A_train, B_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(A_train, B_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(A_train, B_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(A_train, B_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(A_train, B_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


model = models(X_train, Y_train)


from sklearn.metrics import confusion_matrix

for i in range(len(model)):
    cm = confusion_matrix(B_test, model[i].predict(A_test))

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

    print(classification_report(B_test, model[i].predict(A_test)))

    print(accuracy_score(B_test, model[i].predict(A_test)))
    print()







# Print Prediction of Random Forest Classifier model


from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn import datasets

from sklearn.externals import joblib
classifier = KNeighborsClassifier()

joblib.dump(model, "model333.pkl")





classifier = joblib.load("model333.pkl")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


from sklearn.datasets import make_blobs

A, b = make_blobs(n_samples=1683, centers=10, n_features=22, random_state=100)

model = KNeighborsClassifier(metric='minkowski')



model.fit(A, B)

pickle.dump(model, open('model333.pkl','wb'))





model = pickle.load(open('model333.pkl','rb'))
print()
print(model.predict([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
# Import all needed libraries and sublibraries






df = pd.read_csv('data.csv')
df.head(7)


df.shape


df.isna().sum()
df.isnull().sum()



df = df.dropna(axis=1)


df.shape


df['diagnosis'].value_counts()


sns.countplot(df['diagnosis'], label="Count")


df.dtypes




from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()
df.iloc[:, 1] = labelencoder_Y.fit_transform(df.iloc[:, 1].values)
print(labelencoder_Y.fit_transform(df.iloc[:, 1].values))



sns.pairplot(df, hue="diagnosis")



df.head(5)


df.corr()



plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt='.0%')



X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=5)




from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



def models(X_train, Y_train):

    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=5)
    log.fit(X_train, Y_train)


    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state=5)
    svc_lin.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=5)
    svc_rbf.fit(X_train, Y_train)


    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)


    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=2)
    tree.fit(X_train, Y_train)


    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=7)
    forest.fit(X_train, Y_train)

    # print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


model = models(X_train, Y_train)


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







# Print Prediction of Random Forest Classifier model


from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn import datasets

from sklearn.externals import joblib
classifier = RandomForestClassifier()

joblib.dump(model, "model.pkl")





classifier = joblib.load("model.pkl")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=569, centers=10, n_features=30, random_state=7)

model = RandomForestClassifier(criterion='entropy')



model.fit(X, Y)

pickle.dump(model, open('model.pkl','wb'))




model = pickle.load(open('model.pkl','rb'))
print()
print(model.predict([[21.16,23.04,137.2,1404,0.09428,0.1022,0.1097,0.08632,0.1769,0.05278,0.6917,1.127,4.303,93.99,0.004728,0.01259,0.01715,0.01038,0.01083,0.001987,29.17,35.59,188,2615,0.1401,0.26,0.3155,0.2009,0.2822,0.07526]]))














import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


data = pd.read_csv("heart.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(data.head())

standardscalar = StandardScaler()
X = standardscalar.fit_transform(X)
#X = X[:, [2, 9, 11]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1/3)

#classifier = LogisticRegression(random_state=0, solver='lbfgs')
classifier = RandomForestClassifier(n_estimators=50, random_state=0)
'''rfe = RFE(classifier, 3)
res = rfe.fit(X, y)
print(res.n_features_)
print(res.ranking_)
'''
classifier.fit(X_train, y_train)

print(classifier.score(X_test, y_test))

print(confusion_matrix(y_test, classifier.predict(X_test)))
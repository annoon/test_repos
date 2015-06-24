from sklearn.datasets import load_iris
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

iris = load_iris()
'''print iris.data.shape'''

X=iris.data[:,:2]
Y=iris.target
X_train=X[:90]
Y_train=Y[:90]
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)
'''print X_train.shape, Y_train.shape'''

clf=DecisionTreeClassifier()
clf_fitted=clf.fit(X_train,Y_train)
print clf_fitted.score(X_test,Y_test)
scores=cross_validation.cross_val_score(clf,X,Y,cv=5)
print "accuary: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2)

clf=svm.LinearSVC(C=1)
clf_fitted=clf.fit(X_train,Y_train)
print clf_fitted.score(X_test,Y_test)
scores=cross_validation.cross_val_score(clf,X,Y,cv=5)
print "accuary: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2)

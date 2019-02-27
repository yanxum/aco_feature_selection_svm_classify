from sklearn import svm
from sklearn import metrics

def class_svm(train, train_result, test, test_result, gamma, c):
    clf = svm.SVC(kernel='poly', gamma=gamma, C=c)
    clf.fit(train, train_result)
    predict = clf.predict(test)
    clf.get_params(deep = True)
    y = metrics.accuracy_score(test_result, predict)
    return y
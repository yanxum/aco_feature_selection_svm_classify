from feature_select import SVM

def funcFitness(train, train_result, test, test_result,gamma,c):
    fitness = SVM.class_svm(train, train_result, test, test_result,gamma,c)
    return fitness
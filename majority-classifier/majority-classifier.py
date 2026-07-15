import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    results = []
    for c in classes:
        sum=0
        for i in y_train:
            if i==c:
              sum+=1
        results.append(sum)
    idx = np.argmax(results)
    y_test = np.full(X_test.shape,classes[idx])
    return y_test
    pass
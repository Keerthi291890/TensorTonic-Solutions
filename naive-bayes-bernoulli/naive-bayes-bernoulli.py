import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    y_train = np.asarray(y_train)
    X_train = np.asarray(X_train)
    classes = np.unique(y_train)
    class_counts = []
    for c in classes:
        class_counts.append(np.sum(y_train==c))
    class_counts = np.asarray(class_counts)
    priors = class_counts/len(y_train)
    log_priors = np.log(priors)
    # log(P(x/y))
    feature_probs = []
    for c in classes:
        X_c = X_train[y_train==c]
        n_c = X_c.shape[0]
        count = np.sum(X_c,axis=0)
        feature_probs.append((count+1)/(n_c+2))
    feature_probs = np.asarray(feature_probs)
    log_theta = np.log(feature_probs)
    results = []
    for x in X_test:
        samples_scores = []
        for i , c in enumerate(classes):
            score = log_priors[i]
            theta = feature_probs[i]
            for j in range(len(x)):
                score += x[j]*np.log(theta[j]) + (1-x[j])*np.log(1-theta[j])
            samples_scores.append(score)
        results.append(samples_scores)
    return np.array(results)
    pass
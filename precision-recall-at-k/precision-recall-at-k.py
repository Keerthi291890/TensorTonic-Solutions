import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended = np.asarray(recommended)
    relevant = np.asarray(relevant)
    recom = recommended[:k]
    hits = len(set(recom)&set(relevant))
    precision = hits/k
    recall = hits/len(relevant)
    sol = [precision,recall]
    return sol
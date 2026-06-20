import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    result = []
    for seq in seqs:
        if len(seq)>max_len:
           seq = seq[:max_len]
        elif len(seq)<max_len:
            pad_length = max_len-len(seq)
            seq = seq+[pad_value]*pad_length
        result.append(seq)
    return np.array(result)
    pass
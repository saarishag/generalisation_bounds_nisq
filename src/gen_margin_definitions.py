import numpy as np

def per_sample_margin(svm, K_train, y_train):
    """Use noisy alphas(or noisy kernels) in clean decision function"""

    alpha_y = svm.dual_coef_[0] #shape (1, n_support_vectors)

    support_indices = svm.support_ #indices of support vectors
    y_support_vec = y_train[support_indices] #+1/-1 (labels) of the support vectors
    
    alpha = alpha_y/y_support_vec #bc we need alpha only not alpha_i*y_i

    K_sv = K_train[np.ix_(support_indices, support_indices)]
    Y = np.diag(y_support_vec)
    sq_weighted_norm = alpha.T @ Y @ K_sv @ Y @ alpha
    margin = 1/sq_weighted_norm

    f_train = svm.decision_function(K_train)
    func_margin = y_train * f_train #y_i * f_i
    geom_margin = func_margin/np.sqrt(sq_weighted_norm)

    return geom_margin

def corrupt_labels(y, corrupt_lvl):
    """
    Randomly corrupt a fraction of labels"""
    y_corrupted = y.copy()
    n_corrupt = int(corrupt_lvl*len(y)) #fraction*number of labels = integer

    if n_corrupt > 0:
        corrupt_indices = np.random.choice(len(y), n_corrupt, replace=False) #choose n random indices 
        y_corrupted[corrupt_indices] = -y_corrupted[corrupt_indices] #flip the labels
    return y_corrupted



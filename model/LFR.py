from abc import ABC
import numpy as np
import scipy.optimize as optim
from model.utils.utils import *


class LFR(ABC):
    """Learning fair representations is a pre-processing technique that finds a
            latent representation which encodes the data well but obfuscates information
            about protected attributes [R. Zemel, et al.]_.
            References:
                ... [R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,]  "Learning
                   Fair Representations." International Conference on Machine Learning,
                   2013.
            Based on codes from https://github.com/zjelveh/learning-fair-representations
            and https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/lfr.py
            """
    def __init__(self, sensitive_feature, privileged_class, unprivileged_class, seed, output_feature, parameter):
        self.sensitive_f = sensitive_feature
        self.priv_class = privileged_class
        self.unpriv_class = unprivileged_class
        self.outupt_f = output_feature
        self.seed = seed
        self.k = parameter['k']
        self.Ax = parameter['Ax']
        self.Ay = parameter['Ay']
        self.Az = parameter['Az']
        self.max_iter = parameter['max_iter']
        self.max_fun = parameter['max_fun']
        self.w = None
        self.prototypes = None
        self.learned_model = None

    def fit(self, X, y):
        print(">>>>>>>>fit_called<<<<<<<<<<")
        np.random.seed(self.seed)
        idx_priv = np.where(y[self.sensitive_f].values == self.priv_class)[0]
        idx_unpriv = np.where(y[self.sensitive_f].values == self.unpriv_class)[0]
        data_priv = X[idx_priv, :]
        data_unpriv = X[idx_unpriv, :]
        self.features_dim = X.shape[1]
        Y_priv = y[self.outupt_f].values[idx_priv]
        Y_unpriv = y[self.outupt_f].values[idx_unpriv]
        parameter_init = np.random.uniform(size=self.features_dim * 2 + self.k + self.features_dim * self.k)  # w init M init
        Bound = []
        for i, k2 in enumerate(parameter_init):
            if i < self.features_dim * 2 or i >= self.features_dim * 2 + self.k:
                Bound.append((None, None))
            else:
                Bound.append((0, 1))

        self.learned_proto = optim.fmin_l_bfgs_b(lfr, x0=parameter_init, epsilon=1e-5, args=(
            data_priv, data_unpriv, Y_priv, Y_unpriv, self.k, self.Ax,self.Ay,self.Az), bounds=Bound, approx_grad=True, maxfun=self.max_fun,
                                                 maxiter=self.max_iter)[0]

    def transform(self,X, y, threshold = 0.5):
        print(">>>>>>>>>>Transform_called<<<<<<<<<<")
        Y = y.copy()
        idx_priv = np.where(y[self.sensitive_f].values == self.priv_class)[0]
        idx_unpriv = np.where(y[self.sensitive_f].values == self.unpriv_class)[0]
        data_priv = X[idx_priv, :]
        data_unpriv = X[idx_unpriv, :]
        Y_priv = y[self.outupt_f].values[idx_priv]
        Y_unpriv = y[self.outupt_f].values[idx_unpriv]
        Y_hat_p, Y_hat_unp, M_nk_priv, M_nk_unpriv = lfr(
            self.learned_proto,data_priv,data_unpriv,Y_priv,Y_unpriv,results=1)

        # Y_hat_p = (np.array(Y_hat_p)>threshold).astype(np.float64)
        # Y_hat_unp = (np.array(Y_hat_unp)>threshold).astype(np.float64)
        Y[self.outupt_f].iloc[idx_priv] = Y_hat_p.reshape(-1)
        Y[self.outupt_f].iloc[idx_unpriv] = Y_hat_unp.reshape(-1)
        # data_priv[self.outupt_f] = Y_hat_p
        # data_unpriv[self.outupt_f] = Y_hat_unp
        # X.iloc[idx_priv] = data_priv
        # X.iloc[idx_unpriv] = data_unpriv
        X_pred = np.zeros((X.shape[0],M_nk_priv.shape[1]))
        X_pred[idx_priv,:] = M_nk_priv
        X_pred[idx_unpriv,:] = M_nk_unpriv
        return X_pred,Y

    def fit_transform(self, X, y=None, **fit_params):
        pass

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:32:06 2025

@author: talia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from clean_river_data import get_matrix_X
from scipy.linalg import expm


def dynamic_mode_decomposition(X, rank=None):
    """
    Perform Dynamic Mode Decomposition (DMD) using Singular Value Decomposition (SVD).

    Parameters:
    X : numpy.ndarray
        The input data matrix where columns represent snapshots in time.
    rank : int, optional
        The rank for truncation in SVD. If None, no truncation is applied.

    Returns:
    phi : numpy.ndarray
        The DMD modes (spatial modes).
    omega : numpy.ndarray
        The DMD eigenvalues (temporal growth/decay and frequencies).
    b : numpy.ndarray
        The amplitudes of DMD modes.
    """
    # Split the data matrix into X1 and X2
    X1 = X[:, :-1]  # All columns except the last
    X2 = X[:, 1:]   # All columns except the first

    # Perform Singular Value Decomposition on X1
    U, Sigma, VT = np.linalg.svd(X1, full_matrices=False)
    plt.scatter([i for i in range(len(Sigma))],Sigma)
    plt.figure()
    # Truncate if rank is specified
    if rank is not None:
        U = U[:, :rank]
        Sigma = np.diag(Sigma[:rank])
        VT = VT[:rank, :]
    else:
        Sigma = np.diag(Sigma)

    # Compute the reduced A tilde
    A_tilde = np.dot(np.dot(np.dot(U.conj().T,X2),VT.conj().T), np.linalg.inv(Sigma))

    # Eigen decomposition of A_tilde
    mu, W = np.linalg.eig(A_tilde)

    # Compute the DMD modes
    phi = np.dot(np.dot(np.dot(X2,VT.conj().T),np.linalg.inv(Sigma)), W)

    return phi, mu



def predict_dmd(phi, mu, t_future, x0):
    """
	Predict future states using DMD.

    Parameters:
    phi : numpy.ndarray
        The DMD modes (spatial modes).
    mu : numpy.ndarray
        The DMD eigenvalues (temporal growth/decay and frequencies).
    t_future : numpy.ndarray
        Future time points to predict.

    Returns:
    X_future : numpy.ndarray
        Predicted future states.
	"""
	# Compute the temporal dynamics for future time points
    Lambda = np.diag(mu)
    print(np.size(Lambda))
    print(np.size(phi))
    x = np.zeros((len(x0), len(t_future)))
    for i in range(len(t_future)):
        x[:,i] = np.dot(np.dot(np.dot(phi,Lambda**(i+1)),phi.conj().T),x0)
    return x

X_Q, X_F, actual_Q, actual_F, label = get_matrix_X()

# M = matrix being used for DMD (to make changing easier)
M = X_F
a = actual_F

# Normalise the data
scaler = StandardScaler()
M_normalised = scaler.fit_transform(M.T).T  # Normalize rows (features)
M_denorm = scaler.inverse_transform(M.T).T 
#M_normalised =M

# Perform DMD
phi, mu = dynamic_mode_decomposition(M_normalised) #Ignore Rank for now

# Predict future states
num_months = len(M[0]) # Future months: start with first month not found (len(M[0]))
tspan = np.arange(num_months)
n = 20
t_future = np.linspace(num_months, num_months+n, n+1)
X_future = predict_dmd(phi, mu, t_future, M_normalised[:,-1])

#normalise prediction (incorrect method)
X_future_normalised = scaler.fit_transform(np.real(X_future).T).T
X_future = scaler.inverse_transform(np.abs(X_future).T).T
#normalise actual
actual_norm = scaler.fit_transform(a.T).T
actual = scaler.inverse_transform(np.abs(a).T).T
val=8

t = np.linspace(1, num_months, num_months+1)
plt.plot(M_normalised[val])

plt.plot(t_future,X_future_normalised[val])

plt.plot(a[0]-a[0,0]+num_months,actual_norm[val+3])

plt.legend(['Used Data','Predicted Data', 'Actual Data'])
plt.xlabel('time (Months)')
plt.ylabel(f'{label[val+3]} (normalised)')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from solvers import py_ridge, py_lasso_pg, py_lasso_pdhg
from parameterselection import golden_section, gridSearch, DP_secant, DP_bisection,GCV_golden, UPRE_golden

def linearregressionexample():
    N = 1000 # number of points
    M = 100 # Number of dimensions
    M_nonzero = 99 # Number of nonzero coefficients

    # Create regression coefficients
    betas = np.zeros(M)
    betas[:M_nonzero] = np.random.uniform(0.1,1.0,M_nonzero)
    signs = np.random.binomial(1,0.5, M)
    for i,sign in enumerate(signs):
        if sign:
            betas[i] = betas[i]
        else:
            betas[i] = -betas[i]
    # Data matrix X and response Y
    X = np.zeros((N,M)) 
    Y = np.zeros(N)

    # Noise
    sigma = 0.1
    noise_vec = np.random.normal(0.0,sigma,N)
    # Covariance matrix of X
    cov_eigvecs = np.random.uniform(-1.0,1.0,M*M).reshape((M,M))
    cov_eigs_diag = np.ones(M)
    cov_eigs = np.zeros((M,M))
    np.fill_diagonal(cov_eigs, cov_eigs_diag)
    cov = cov_eigvecs.T @ cov_eigs @ cov_eigvecs
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    corr = cov / outer_v
    corr[cov == 0] = 0
    chol = sp.linalg.cholesky(corr, lower=True) 

    X = np.random.normal(0, 1.0,M*N).reshape((N,M))
    X = np.dot(X,chol)
    Y = np.dot(X[:,:M_nonzero],betas[:M_nonzero]) + noise_vec

    XX = np.matmul(X.T, X)
    lam_l = 0.2
    lam_r = 1.0 - lam_l
    #u_true = u_pg = py_lasso_pg(X,Y,lam_l,lam_r, maxiter = 20000, tol = 1e-12)
   
    u_GCV, t_GCV  = GCV_golden(X,Y,py_ridge, method = 'eigh')
    u_UPRE, t_UPRE  = UPRE_golden(X,Y,py_ridge, method = 'eigh')
    t_true, _ = gridSearch(X,Y, betas, 1000,py_ridge)
    print(t_GCV, t_UPRE, t_true)
    exit()
    u_ridge = py_ridge_cholesky(X,Y,t_ridge,1-t_ridge)
    t_lasso_hat, R_hat = gridSearch(X,Y, u_ridge, 1000, py_lasso_pg, tol = 1e-8,maxiter = 1000)
    t_lasso_true, R = gridSearch(X,Y, betas, 1000, py_lasso_pg, tol = 1e-8,maxiter = 1000)
    plt.semilogy(R_hat, label = 'hat')
    plt.semilogy(R, label = 'true')
    plt.legend()
    print(t_lasso_hat - t_lasso_true)
    plt.show()
    linear = py_ridge_cholesky(X,Y,1.0,0.0)
    print(np.linalg.norm((betas - linear)))
    print(np.linalg.norm((betas - u_ridge)), t_ridge)
    #u_lasso_true = py_lasso_pg(X,Y,t_lasso_true,(1 - t_lasso_true), tol = 1e-8, maxiter = 10000) 
    #u_lasso_hat = py_lasso_pg(X,Y,t_lasso_true,(1 - t_lasso_hat), tol = 1e-8, maxiter = 10000) 
    u_lasso_true, t_lasso_true = golden_section(X,Y,betas,py_lasso_pg, tol = 1e-8, maxiter = 10000)
    u_lasso_hat, t_lasso_hat = golden_section(X,Y,u_ridge,py_lasso_pg, tol = 1e-8, maxiter = 10000)
    print(np.linalg.norm((betas - u_lasso_hat)), t_lasso_hat)
    print(np.linalg.norm((betas - u_lasso_true)), t_lasso_true)
    print(u_lasso_hat)
    print(u_lasso_true)

linearregressionexample()

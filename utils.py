from termcolor import colored
import time
from operator import itemgetter
from scipy import sparse, linalg, signal, misc, fftpack, ndimage
from scipy.linalg import circulant, toeplitz
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
from PIL import Image
from scipy.stats import ortho_group

def unit_vector(vector):
    """ 
    Normalises a vector  
    """
    return vector / np.linalg.norm(vector)

def project_them(Q, X):
    """ 
    Projecting to a subspace given by orthonormal rows of a matrix Q 
    """
    P = np.dot(Q.T, Q)
    return np.dot(P, X.T).T

def safe_norm_div(x, y):
    """ 
    Calculates ||x-y||/||y|| safely 
    """
    if (np.linalg.norm(y) == 0.0):
        return 0.0
    return np.linalg.norm(x - y) / np.linalg.norm(y)

def rank_r(m, d, r):
    """
    create a unit norm matrix of a certain rank
    """
    U, ss, V = np.linalg.svd(np.random.randn(m, d))
    idx = np.sort(np.random.choice(min(m, d), r, replace = False))
    S = np.zeros((m, d))
    S[idx, idx] = ss[idx]
    A = np.dot(U, np.dot(S, V))
    A /= np.linalg.norm(A, 2)
    return A

def set_it_up(m, d, rank = False, h = 5, N_train = 50, N_test = 1, sigma = 0.05):
    """
    Create the matrix A of size m x d, and a certain rank
    Create N_train training data X_i^train, Y_i^train, with noise level sigma
    Create N_test testing data X_i^test, Y_i^test
    """

    # Create the matrix A
    if rank == False:
        A = np.random.randn(m ,d)
        A /= np.linalg.norm(A, 2)
    else:
        A = rank_r(m, d, rank)

    # Create the training data
    mean_X = np.zeros(d)
    cov_X = np.zeros((d, d))
    cov_X[range(h), range(h)] = 1.0 # set the data dimensionality
    X = np.random.multivariate_normal(mean_X, cov_X, N_train)

    # Noise
    mean_eta = np.zeros(m)
    cov_eta = np.eye(m)
    Eta = sigma * np.random.multivariate_normal(mean_eta, cov_eta, N_train)

    # Create Y_i = A X_i + Eta_i
    Y = np.zeros((N_train, m))
    for i in range(N_train):
        Y[i, :] = np.dot(A, X[i, :]) + Eta[i, :]

    ## Create the testing data
    X_test = np.random.multivariate_normal(mean_X, cov_X, N_test)
    Eta_test = sigma * np.random.multivariate_normal(mean_eta, cov_eta, N_test)
    Y_test = np.zeros((N_test, m))
    for i in range(N_test):
        Y_test[i, :] = np.dot(A, X_test[i, :]) + Eta_test[i, :]

    ## Create empirical projection from the training data
    piY_n = pi_hat_n(covariance(Y), h = h)

    return A, piY_n, X_test, Y_test

def set_it_up_id(n,  h = 5, N_train = 50, N_test = 1, sigma = 0.05):
    """
    MARTIN
    Same as set_it_up but uses A = id
    Also "spreads" the nonzero elements of the covariance matrix
    """
    # Create the training data
    mean_X = np.zeros(n)
    ones = np.zeros(n)
    ones[range(h)] = 1.0
    diag = np.random.choice(ones,n,False)
    cov_X = np.diag(diag) # set the data dimensionality
    X_train = numpy.random.multivariate_normal(mean_X, cov_X, N_train)

    # Noise
    mean_eta = np.zeros(n)
    ones = np.ones(n)
    cov_eta = np.diag(ones)
    Eta_train = sigma * numpy.random.multivariate_normal(mean_eta, cov_eta, N_train)

    # Create Y_i = X_i + Eta_i
    Y_train = np.zeros((N_train, n))
    Y_train = X_train + Eta_train

    # Create the testing data
    X_test = np.random.multivariate_normal(mean_X, cov_X, N_test)
    Eta_test = sigma * np.random.multivariate_normal(mean_eta, cov_eta, N_test)
    Y_test = np.zeros((N_test, n))
    Y_test = X_test + Eta_test

    # Create empirical projection from the training data
    piY_n = pi_hat_n(covariance(Y_train), h = h)

    return piY_n, X_test, Y_test

def create_empirical_estimator_without_resize(image_array, amount, shape, h, N_train, N_test, sigma = 0.1):
    N_image = len(image_array)
    dtype = 'float64'
    X_train = np.empty((N_train,) + shape, dtype)
    X_test = np.empty((N_test,) + shape, dtype)
    Y_train = np.empty((N_train,np.prod(shape)), dtype)
    Y_test = np.empty((N_test,) + shape, dtype)

    ii = 0
    for i in range(N_image):
        for j in range(amount[i]):
            X_train[ii+j,:,:] = np.copy(image_array[i, :, :])
        ii += amount[i]
    for i in range(N_train):
        Y_train[i,:] = X_train[i,:,:].reshape(np.prod(shape)) + sigma * np.random.normal(0,1,np.prod(shape))
    X_test[0,:,:] = np.copy(image_array[0, :, :])
    Y_test[0,:, :] = X_test[0,:,:] + sigma * np.random.normal(0,1,np.prod(shape)).reshape(shape) 

    piY_n = pi_hat_n(covariance(Y_train), h = h)

    return piY_n, X_test, Y_test

def create_empirical_estimator(image_array, newshape, h, N_train, N_test, sigma = 0.1, calc = True):
    N_image = len(image_array)
    dtype = 'float64'
    origshape = image_array[0].shape
    u_train = np.empty((N_train+1,) + origshape, dtype)
    #u_test = np.empty((N_test,) + origshape, dtype)
    f_train = np.empty((N_train+1,) + origshape, dtype)
    f_train_resized = np.empty((N_train+1,np.prod(newshape)), dtype)
    #f_test = np.empty((N_test,) + origshape, dtype)
    for i in range(N_train + 1):
        u_train[i,:,:] = np.copy(image_array[0, :, :])
    for i in range(N_train + 1):
        f_train[i,:,:] = np.minimum(np.maximum(u_train[i,:,:] + sigma * np.random.normal(0,1,np.prod(origshape)).reshape(origshape), 0.0), 1.0)
        f_train_image = Image.fromarray(f_train[i])
        f_train_image = f_train_image.resize(newshape, Image.BICUBIC)
        f_train_resized[i] = np.array(f_train_image).reshape(np.prod(newshape))
    u = u_train[0]
    f = f_train[0]
    print("CREATING PROJECTION MATRIX")
    t = time.time()
    if (calc == True):
        S = np.loadtxt("cov.txt")
        Pi = pi_hat_n(S, h = h)
    else:
        Pi = pi_hat_n(covariance(f_train_resized), h = h)
    u_hat = np.dot(f_train_resized[0], Pi).reshape(newshape)
    u_hat_image = Image.fromarray(u_hat)
    u_hat_image = u_hat_image.resize(origshape, Image.BICUBIC)
    u_hat = np.array(u_hat_image)
    print("TIME TO CALCULATE u_hat:", time.time() - t)
    return u, f, u_hat

def create_empirical_estimator_with_dataset():
    return 0

def create_empirical_estimator_alt(image_array, newshape, h, N_train, N_test, sigma = 0.1, calc = True):
    N_image = len(image_array)
    print(N_image)
    dtype = 'float64'
    origshape = image_array[0].shape
    u_train = np.empty((N_train+1,) + origshape, dtype)
    #u_test = np.empty((N_test,) + origshape, dtype)
    f_train = np.empty((N_train+1,) + origshape, dtype)
    f_train_resized = np.empty((N_train+1,np.prod(newshape)), dtype)
    #f_test = np.empty((N_test,) + origshape, dtype)
    for i in range(N_train):
        u_train[i,:,:] = np.copy(image_array[i, :, :])
    for i in range(N_train):
        f_train[i,:,:] = u_train[i,:,:] + sigma * np.random.normal(0,1,np.prod(origshape)).reshape(origshape)
        f_train_image = Image.fromarray(f_train[i])
        f_train_image = f_train_image.resize(newshape, Image.LANCZOS)
        f_train_resized[i] = np.array(f_train_image).reshape(np.prod(newshape))
    u = u_train[671]
    f = f_train[671]
    print("CREATING PROJECTION MATRIX")
    t = time.time()
    if (calc == True):
        S = np.loadtxt("cov.txt")
        Pi = pi_hat_n(S, h = h)
    else:
        Pi = pi_hat_n(covariance(f_train_resized), h = h)

    Pi = pi_hat_n(covariance(f_train), h = h)
    #u_hat = np.dot(f_train_resized[671], Pi).reshape(newshape) 
    u_hat = np.dot(f_train[671], Pi).reshape(newshape)
    #u_hat_image = Image.fromarray(u_hat)
    #u_hat_image = u_hat_image.resize(origshape, Image.LANCZOS)
    #u_hat = np.array(u_hat_image)
    print("TIME TO CALCULATE u_hat:", time.time() - t)
    return u, f, u_hat

def empirical_estimators(A, Y, Pi_n):
    """
    for a given A, Y compute the empirical estimator

    """
    N = Y.shape[0]
    X = np.zeros((N, A.shape[1]))
    Eta = np.zeros((N, A.shape[0]))
    Ainv = np.linalg.pinv(A)

    for i in range(N):
        Pi_n_Y = np.dot(Pi_n, Y[i, :])
        X[i, :] = np.dot(Ainv, Pi_n_Y)
        Eta[i, :] = Y[i, :] - Pi_n_Y

    return X, Eta

    """
    soft thresholding a vector with a threshold being lam
    TODO: replace with inbuilt func
    """
    s_tv = np.zeros(vec.shape)
    # import pdb; pdb.set_trace()
    for i in range(len(vec)):
        if vec[i] > lam / 2.0:
            s_tv[i] = vec[i] - lam / 2.0
        elif vec[i] < - lam / 2.0:
            s_tv[i] = vec[i] + lam / 2.0
    return s_tv

def sgn(v):
    """
    return sign of a vector
    """
    return np.sign(v)

def covariance(Z):
    """
    straightforward computation of the empirical covariance
    """
    N = Z.shape[0]
    S = np.zeros((Z.shape[1], Z.shape[1]))
    for i in range(N):
        S += 1.0 / N * np.outer(Z[i, :], Z[i, :])
    return S

def covariance_matrix(Z):
    N = Z.shape[0]
    S = np.zeros((Z.shape[1]*Z.shape[2], Z.shape[1]*Z.shape[2]))

    for i in range(N):
        S += 1.0 / N * np.outer(Z[i, :, :], Z[i, :,:])
        print(i)
    return S

def pi_hat_n(S, h = 1):
    """
    compute the empirical projection of a given rank h
    """
    #np.savetxt("cov.txt", S) 
    n = S.shape[0] 
    #plt.spy(S)
    #plt.show()
    #plt.imsave("COV.png", S, cmap = "gray")
    eigv, eigvectors = linalg.eigh(S, eigvals = (n-h,n-1))
    print(eigv)
    #eigv, eigvectors = linalg.eigh(S)
    plt.semilogy((eigv[::-1])[:h])
    plt.xlabel("Eigenvalue number")
    plt.ylabel("Eigenvalue")
    plt.show()
    #print("Found eigvals")
    eigvectors = eigvectors[:,-h:].T[::-1].T
    #eigvectors = eigvectors[:, -h:].T[::-1].T
    P = eigvectors.dot(eigvectors.T)
    return P

def add_gaussian_noise(f, sigma=0.001):
    """
    Adds gaussian noise to image
    Assumes image is rescaled to lie between 1.0 and 0.0.
    """
    out = np.zeros((2,) + f.shape, f.dtype)

    shape = f.shape

    out = np.minimum(np.maximum(f + sigma* numpy.random.normal(0,1,np.prod(shape)).reshape(shape), 0.0), 1.0)
    return out

def add_gaussian_blurring(f, ker, sigma):
    out = signal.fftconvolve(f,ker, mode='same')
    return out

def grad(f):
    """
    Calculates gradient of image f of size n,m
    returns gradient of size 2,n,m

    """
    out = np.zeros((2,) + f.shape, f.dtype)

    # x-direction
    out[0, :-1, :] = f[1:, :] - f[:-1, :]

    # y-direction
    out[1, :, :-1] = f[:, 1:] - f[:, :-1]
    return out

def better_grad(f):
    out = np.zeros((2,) + f.shape, f.dtype)

    # x-direction
    out[0,0, :] = f[1,:] - f[0, :]
    out[0,-1, :] = f[-1,:] - f[-2, :]
    out[0, 1:-1, :] = (f[2:, :] - f[:-2, :])/2

    # y-direction
    out[0,:, 0] = f[:, 1] - f[:, 0]
    out[0,:, -1] = f[:,-1] - f[:, -2]

    out[1, :, 1:-1] = (f[:, 2:] - f[:, :-2])/2
    return out

def div(f):
    """
    Calculates divergence of image f of size 2,n,m
    returns divergence of size n,m
    """
    out = np.zeros_like(f)

    # Boundaries along y-axis
    out[0, 0, :] = f[0, 0, :]
    out[0, -1, :] = -f[0, -2, :]
    # Inside along y-axis
    out[0, 1:-1, :] = f[0, 1:-1, :] - f[0, :-2, :]

    # Boundaries along y-axis
    out[1, :, 0] = f[1, :, 0]
    out[1, :, -1] = -f[1, :, -2]
    # Inside along y-axis
    out[1, :, 1:-1] = f[1, :, 1:-1] - f[1, :, :-2]

    # Return sum along x-axis
    return np.sum(out, axis=0)
def norm1(f, axis=0, keepdims=False):
    """
    returns 1-norm of image f of size n,m
    returns number
    """
    return np.sqrt(np.sum(f**2, axis=axis, keepdims=keepdims))

def better_norm1(f):
    return np.sqrt(f[0,:,:]**2 + f[1,:,:]**2)

def TV(u):
    return np.sum(better_norm1(grad(u)))

def anisotropic_TV(u):
    return np.sum(np.abs(u[1:,:] - u[:-1,:])) + np.sum(np.abs(u[:,1:] - u[:,:-1]))

def cost(u,f,lam):
    return 0.5* np.linalg.norm(u-f) + lam*TV(u) 

def cost_matrix(A,u,f,lam):
    return 0.5* np.linalg.norm(A.dot(u.reshape(f.shape[0]*f.shape[1])).reshape(f.shape)-f) + lam*TV(u) 

def psnr(noisy,true):
    """
    Calculates psnr between two images
    """
    MSE = np.mean((noisy-true)**2)
    MAX = max(np.max(noisy),np.max(true))
    return 10 * np.log10(MAX**2/MSE)
    #return 20*np.log10(mse) - 10*np.log10(np.max(np.max(noisy),np.max(true)))

def dualitygap_denoise(lam,u,divp,f):
    return 0.5 * np.linalg.norm(u - f)**2 + lam * np.sum(better_norm1(grad(u))) + 0.5 * np.linalg.norm(f + divp)**2 - 0.5 * np.linalg.norm(f)**2
    #return 0.5 * np.linalg.norm(u - f)**2 + lam * np.sum(better_norm1(grad(u))) - 0.5 * np.linalg.norm(f - divp)**2 + 0.5 * np.linalg.norm(f)**2 + np.linalg.norm(divp)**2



import numpy as np
import matplotlib.pyplot as plt
import time

from utils import *
from ChambollePock import *
#from ChambollePock_jit import ChambollePock_denoise_jit
from projectedGradient import *

def discrepancy_ruleTV(f, noise_std, tau = 1.0, lam_init= 2.0, q = 0.9):
    """
    Implementation of the discrepancy principle
    noise_std is the estimate (or the true value) of the noise parameter
    """
    nm = np.prod(f.shape)
    i = 0
    lam = lam_init
    max_iter = 50
    error_level = tau * noise_std * np.sqrt(nm)
    tol = 1e-5
    while (i < max_iter):
        u = ChambollePock_denoise(f,lam, tau = 0.5, sig = 0.25, acc = True, tol = 1.0e-5)
        if np.linalg.norm(u - f) < error_level:
            break
        if (lam <= tol):
    	    break
        i += 1
        lam = lam_init * (q ** i)
    t = 1.0 / (1.0 + lam_init * (q ** i))
    return u, t

def quasi_optimalityTV(f, lam_init = 2.0, q = 0.9):
    """
    Implementation of quasi optimality for parameter selection
    """
    
    lam = lam_init
    max_iter = 50
    error = np.zeros(max_iter)
    #alt_error = np.zeros(max_iter)
    u_old = ChambollePock_denoise(f,lam, tau = 0.5, sig = 0.25, acc = True, tol = 1.0e-5)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = ChambollePock_denoise(f,lam, tau = 0.5, sig = 0.25, acc = True, tol = 1.0e-5)
        error[i]  = np.linalg.norm(u_old - u_new)
        #alt_error[i] = np.linalg.norm(u_old - u_new) /abs(lam_init*(q ** i - q ** (i-1)))
        u_old = np.copy(u_new)

    #plt.plot(error)
    #plt.plot(alt_error)
    #plt.show()
    opt_idx = np.argmin(error[error != 0.0])
    t = 1.0 / (1.0 + lam_init * (q ** opt_idx))
    lam = lam_init * (q ** opt_idx)
    u= ChambollePock_denoise(f,lam, tau = 0.5, sig = 0.25, acc = True, tol = 1.0e-5)
    
    return u, t

def L_curveTV(f,lam_init = 2.0, q = 0.9):
    """
    Implementation of the L-curve method
    """
    lam = lam_init
    max_iter = 50
    residual_list = np.zeros(max_iter)
    size_list = np.zeros(max_iter)
    error = np.zeros(max_iter)
    alt_error = np.zeros(max_iter)
    
    for i in range(max_iter): #range(max_iter):
        u = ChambollePock_denoise(f,lam, tau = 0.5, sig = 0.25, acc=True, tol = 1.0e-5)
        #u, _, j = projected_gradient_alt(f,lam, tau = 0.2, tol = 1.0e-4)
        lam = lam_init * (q ** i)
        residual_list[i] = np.linalg.norm(u - f)
        size_list[i] = np.linalg.norm(u)
        error[i]  = np.linalg.norm(u - f) * np.linalg.norm(u)
    #plt.loglog(residual_list,size_list)
    #plt.show()
    opt_idx = np.argmin(error)
    t = 1.0 / (1.0 + lam_init * (q ** opt_idx))
    lam = lam_init * (q ** opt_idx)
    u = ChambollePock_denoise(f,lam, tau = 0.5, sig = 0.25, acc = True, tol = 1.0e-5)
    return u, t

def R(t, f, u):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t = ChambollePock_denoise(f, (1-t)/t, tau = 0.25, sig = 0.25, acc = True, tol = 1e-4)
    return np.linalg.norm(u_t - u)

def R_convolution(t,f,u,fker,fker_star,shape):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t = ChambollePock_convolution_edge(f, (1-t)/t,fker,fker_star, shape,tau = 0.25, sig = 0.25, tol = 1e-4)
    return np.linalg.norm(u_t - u)


def R_basis(t, f, basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.25, acc = True, tol = 1e-4)
    return np.sum([np.sum(np.multiply(basis_img[i],u_t))**2 for i in range(basis_img.shape[0])]) 

def R_basis_vec(t, f, basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.2, acc = True, tol = 1e-2)
    shape = u_t.shape
    u_t = u_t.reshape(np.prod(shape)) 
    print(np.inner(basis_img, u_t)**2)
    return np.sum(np.inner(basis_img, u_t)**2) 

def R_basis_alt(t,f,avg_img,basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.2, acc = True, tol = 1e-2)
        u_t = u_t - avg_img
    return np.linalg.norm(u_t - np.sum([np.sum(np.multiply(basis_img[i],u_t))*basis_img[i] for i in range(basis_img.shape[0])]))**2


def R_basis_vec_alt(t,f,basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.2, acc = True, tol = 1e-2)
    shape = u_t.shape
    u_t = u_t.reshape(np.prod(shape)) 
    return np.linalg.norm(u_t - np.sum([np.inner(basis_img[i],u_t)*basis_img[i] for i in range(basis_img.shape[0])]))

def gridSearch(f, u, N, plot=True):
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    for i in range(0,N):
        R_list[i] = R(t_list[i],f,u)
        print(i, t_list[i], R_list[i])
        #R_listhat[i] = R2D(t_list[i],Y_test,X_hat)
    t_opttrue = t_list[np.argmin(R_list[1:N-1])]
    #t_opthat = t_list[np.argmin(R_listhat[1:N-1])]
    if plot:
        plt.plot(t_list[1:N-1],R_list[1:N-1], color= "red")
        plt.plot(t_opttrue,np.min(R_list[1:N-1]), 'ro')
        #plt.plot(t_list[1:N-1],R_listhat[1:N-1],label="Empirical", color = "blue")
        #plt.plot(t_opthat,np.min(R_listhat[1:N-1]), 'bo')
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.legend()
        plt.grid()
        plt.show()
    #print("optimal lambda", (1 - t_opttrue)/t_opttrue)
    return t_opttrue, R_list[1:N-1]

def gridSearch_convolution(f, u, N, fker, fker_star, shape,plot=True):
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    for i in range(0,N):
        R_list[i] = R_convolution(t_list[i],f,u,fker,fker_star,shape)
        print(i, t_list[i], R_list[i])
        #R_listhat[i] = R2D(t_list[i],Y_test,X_hat)
    t_opttrue = t_list[np.argmin(R_list[1:N-1])]
    #t_opthat = t_list[np.argmin(R_listhat[1:N-1])]
    if plot:
        plt.plot(t_list[1:N-1],R_list[1:N-1], color= "red")
        plt.plot(t_opttrue,np.min(R_list[1:N-1]), 'ro')
        #plt.plot(t_list[1:N-1],R_listhat[1:N-1],label="Empirical", color = "blue")
        #plt.plot(t_opthat,np.min(R_listhat[1:N-1]), 'bo')
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.legend()
        plt.grid()
        plt.show()
    #print("optimal lambda", (1 - t_opttrue)/t_opttrue)
    return t_opttrue, R_list[1:N-1]

def gridSearch_basis(f, basis_img, avg_img, N):
    R_curr = lambda t: R_basis_alt(t,f,avg_img, basis_img)
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    for i in range(0,N):
        R_list[i] = R_curr(t_list[i])
        print(i, t_list[i], R_list[i])
    t_opttrue = t_list[np.argmin(R_list[1:N-1])]
    return t_opttrue, t_list[1:N-1], R_list[1:N-1] 

def gridSearch_basis_vec(f, basis_img, N):
    R_curr = lambda t: R_basis_vec(t,f,basis_img)
    t_list = np.linspace(0.85, 1.0, N)
    R_list = np.zeros(N)
    for i in range(0,N):
        R_list[i] = R_curr(t_list[i])
        print(i, t_list[i], R_list[i])
    t_opttrue = t_list[np.argmin(R_list[1:N-1])]
    return t_opttrue, t_list[1:N-1], R_list[1:N-1]


def optTV(Y, X_hat, tol = 1e-5):
    """
    Implementation of backtracking line search with interpolation
    """
    
    h = - 0.001
    satisfied = False
    cnt = -1
    max_iter = 100
    beta = 0.8
    go = True
    alpha0 = 0.01
    tau = 0.45
    c1 = 1e-2
    t = np.zeros(max_iter + 1); t[0] = 0.99
    t_current = t[0]
    
    descent_directions = np.zeros(len(t))
    loss_functional_evals = np.zeros(len(t))
    alphas = np.zeros(len(t))
    
    while satisfied == False:
        # Compute gradient at t
        cnt += 1
        z, _ = ChambollePock_denoise(Y,(1 - t_current)/t_current)
        t_direction = t_current + h
        z_h, _ = ChambollePock_denoise(Y,(1 - t_direction)/t_direction)
        descent_directions[cnt] =  (np.linalg.norm(z - X_hat) - np.linalg.norm(z_h - X_hat)) / h
        loss_functional_evals[cnt] =  np.linalg.norm(z - X_hat)
    
        # Check stopping criteria
        if (np.abs(descent_directions[cnt]) < tol) or (cnt >= max_iter):
        	satisfied = True
        	break
    
        # Find the step-size
        alpha = np.copy(alpha0)
        tau = - np.abs(descent_directions[cnt]) ** 2
    
        if cnt == 0 :
            last_alpha = alpha
        t_temp = t_current + alpha * descent_directions[cnt]
        phi0 = loss_functional_evals[cnt]
        phiprime0 = tau
        z_temp, _ = ChambollePock_denoise(Y, (1 - t_temp) / t_temp)
        phialpha = np.linalg.norm(z_temp - X_hat)
    
        if (phialpha - phi0) < (c1 * phiprime0 * alpha):
            step_size = alpha
            # print "here and {0}, {1}, {2}, with finally {3}".format(phialpha - phi0, phi0, alpha, c1 * phiprime0 * alpha)
        else:
            step_size = - (alpha ** 2 * phiprime0) / (2.0 * (phialpha - phi0 - phiprime0 * alpha))
            # print "there {0}".format(step_size)
        if (np.abs(last_alpha / step_size) > 10.0) or (np.abs(step_size) < 1e-3):
            # print "{0} and {1}".format(np.abs(step_size - last_alpha), np.abs(step_size / last_alpha))
            step_size = last_alpha * beta * 1.3
    
        last_alpha = np.copy(step_size)
        t_new = t_current + step_size * descent_directions[cnt]
        t[cnt + 1] = t_new
        t_current = t_new
        print("cnt is ", cnt)
    return t_new

def optTV_golden(f, u_hat, t_left = 0.0, t_right = 1.0, tol = 1e-5):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    R_left = R(t_left_new, f, u_hat)
    R_right = R(t_right_new, f, u_hat)

    i = 0
    while (i < max_iter):
        print(t_left, t_right)
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            R_left = R(t_left_new, f, u_hat)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            R_right = R(t_right_new, f, u_hat)
        if (h < tol):
            t_opt = (t_right_new + t_left_new)/2

            u = ChambollePock_denoise(f,(1-t_opt)/t_opt,tau = 0.5, sig = 0.25, acc = True, tol = 1.0e-5)

            return u, t_opt
        i += 1

def optTV_golden_convolution(f, u_hat, fker, fker_star,shape, t_left = 0.0, t_right = 1.0, tol = 1e-5):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2
    
    R_ = lambda t, f, u: R_convolution(t,f,u,fker,fker_star,shape)

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    R_left = R_(t_left_new, f, u_hat)
    R_right = R_(t_right_new, f, u_hat)

    i = 0
    while (i < max_iter):
        print(t_left, t_right)
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            R_left = R_(t_left_new, f, u_hat)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            R_right = R_(t_right_new, f, u_hat)
        if (h < tol):
            t_opt = (t_right_new + t_left_new)/2

            u = ChambollePock_convolution(f,(1-t_opt)/t_opt, fker, fker_star, shape, tau = 0.5, sig = 0.25) 

            return u, t_opt
        i += 1

import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from utils import *
#from numpy.fft import fftn

def R(prob,v,t,solver,u_hat,u0=None,Av=None,loss = None, **solver_kwargs):
    u = solver(prob,v,t,1-t,u0=u0,Av=Av,**solver_kwargs)
    if loss == "alt":
        return u, np.linalg.norm(np.dot(prob.A,u - u_hat))**2
    elif loss == "projected":
        return u, np.linalg.norm(np.dot(prob.P,u) - u_hat)**2
        #return u, np.linalg.norm(prob.P(u) - u_hat)**2
    else:
        return u, np.linalg.norm(u - u_hat)**2

def grid_search(prob, v, solver, u_hat,u_true = None, N = 200, plot=False, loss = None,**solver_kwargs):
    t_list = np.linspace(0.9, 1.0-1e-10, N)
    t_list = t_list[::-1]
    R_list = np.zeros(N)
    E_list = np.zeros(N)
    S_list = np.zeros(N)
    if prob.solve_method == "identity":
        Av = v
    elif prob.solve_method == "cg":
        Av = prob.Aast(v)
    else:
        Av = np.dot(prob.A.T, v)
    

    u, R_list[0] = R(prob,v,t_list[0],solver,u_hat,Av=Av,loss = loss, **solver_kwargs)
    #u, E_list[0] = R(prob,v,t_list[0],solver,u_true,Av=Av, loss = loss, **solver_kwargs)
    #S_list[0] = 2 * np.sum(np.multiply(u, u_true - u_hat)) # Inner product
    for i in range(1,N):
        print(i)
        u, R_list[i] = R(prob,v,t_list[i],solver,u_hat,u0=u,Av=Av,loss = loss,**solver_kwargs)
        #u, R_list[i] = R(prob,v,t_list[i],solver,u_hat,Av=Av,loss = loss,**solver_kwargs)
        #u, E_list[i] = R(prob,v,t_list[i],solver,u_true,u0=u,Av=Av,loss = loss, **solver_kwargs)
        #S_list[i] = 2* np.sum(np.multiply(u, u_true - u_hat))
    t_opt = t_list[np.argmin(R_list)]
    if plot:
        plt.plot(t_list,R_list, color= "red", label = r"$\hat{R}$")
        plt.plot(t_list,E_list, color= "green", label = r"$R$")
        plt.plot(t_list,S_list, color= "blue", label = r"$\langle S(\lambda)v,u - \hat{u} \rangle$")
        #plt.plot(t_opt,np.min(R_list[1:N-1]), 'ro')
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.legend()
        plt.grid()
        plt.show()
    u = solver(prob,v,t_opt,1-t_opt, Av = Av, u0 = u,**solver_kwargs)
    #u = solver(prob,v,t_opt,1-t_opt, Av = Av, **solver_kwargs)
    return u, t_opt

def opt(prob,v,solver,u_hat,noise_std = None,loss = None, tol = 1e-5):
    if prob.solve_method == "identity":
        Av = v
    elif prob.solve_method == "cg":
        Av = prob.Aast(v)
    else:
        Av = np.dot(prob.A.T, v)
    R_ = lambda t, u0: R(prob,v,t,solver, u_hat = u_hat, u0 = u0, loss = loss, Av = Av)
    return golden_section(R_, gstol = tol)

def GCV_func(prob,v,t,solver,u0=None,Av=None,**solver_kwargs):
    """
    For Ridge 
    """
    u = solver(prob,v, t, 1-t,u0 = u0, Av = Av,**solver_kwargs) 
    if prob.solve_method == 'eigh':
        trace = np.sum(t * prob.eigvals/(t * prob.eigvals + 1.0 - t)) 
    elif prob.solve_method == 'svd':
        trace = np.sum(t * prob.shs/(t * prob.shs + 1.0 - t))
    return u, np.linalg.norm(v - np.dot(prob.A,u), 2)**2/(1.0 - trace/np.prod(v.shape))**2

def GCV(prob,v,solver,u_hat = None,noise_std = None,tol=1e-7):
    Av = np.dot(prob.A.T,v)
    GCV_func_ = lambda t, u0: GCV_func(prob,v,t,solver, u0 = u0, Av = Av)
    return golden_section(GCV_func_, gstol = tol)

def UPRE_func(prob,v,t,solver,noise_std,u0=None,Av=None,**solver_kwargs):
    """
    For Ridge 
    """
    u = solver(prob,v,t,1.0-t, **solver_kwargs) 
    if prob.solve_method == 'eigh':
        trace = np.sum(t * prob.eigvals/(t * prob.eigvals + 1.0 - t)) 
    elif prob.solve_method == 'svd':
        trace = np.sum(t * np.abs(prob.shs)**2/(t * np.abs(prob.shs)**2 + 1.0 - t))
    return u, np.linalg.norm(np.dot(prob.A,u) - v)**2 + 2 * noise_std * trace

def UPRE(prob,v,solver,u_hat = None,noise_std = 0.0,tol=1e-7):
    Av = np.dot(prob.A.T,v)
    UPRE_func_ = lambda t, u0: UPRE_func(prob,v,t,solver, noise_std,u0 = u0, Av = Av)
    return golden_section(UPRE_func_, gstol = tol)

def LC_func(prob,v,t,solver, u0 = None, Av=None,**solver_kwargs):
    u = solver(prob,v,t,1.0-t, **solver_kwargs)
    return u, np.linalg.norm(np.dot(prob.A,u) - v)*np.linalg.norm(u)

def LC(prob,v,solver,u_hat=None,noise_std=0.0, tol = 1e-7):
    Av = np.dot(prob.A.T,v)
    LC_func_ = lambda t, u0: LC_func(prob,v,t,solver,u0 = u0,Av = Av)
    return golden_section(LC_func_, gstol = tol)

def golden_section(func, t_left = 0.0, t_right = 1.0 - 1e-10, gstol = 1e-4, gsmax_iter = 1000):
    """
    Golden section search for optimizing functions of one variable

    Parameters  :
        func    : Function to optimize. Should take in t and u0.  

    """

    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2
    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, f_left = func(t_left_new, u0 = None)
    u, f_right = func(t_right_new, u0 = None)

    for i in range(gsmax_iter):
        if (f_left < f_right):
            t_right = t_right_new
            t_right_new = t_left_new
            f_right = f_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, f_left = func(t_left_new, u0 = u)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            f_left = f_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, f_right = func(t_right_new, u0 = u) 
        if (h < gstol):
            t_opt = (t_right_new + t_left_new)/2

            u, _ = func(t_opt, u0 = u)

            return u, t_opt
def optTV(f, u_hat, solver, t_left = 0.5, t_right = 0.99, optTVtol = 1e-4, **solver_kwargs):
    """
    CHANGED SO THAT solver TAKES IN t, NOT LAMBDA
    """
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, R_left = R(t_left_new, f, u_hat, solver, **solver_kwargs)
    u, R_right = R(t_right_new, f, u_hat, solver, **solver_kwargs)

    i = 0
    while (i < max_iter):
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, R_left = R(t_left_new, f, u_hat, solver, u0 = u, **solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, R_right = R(t_right_new, f, u_hat, solver, u0 = u,**solver_kwargs)
        if (h < optTVtol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(f, (1-t_opt)/t_opt, **solver_kwargs)

            return u, t_opt
        i += 1

def optTV_convolution(f, u_hat, fker, fker_star,shape, solver,t_left = 0.0, t_right = 1.0, tol = 1e-6, **solver_kwargs):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2
    
    R_ = lambda t, f, u: R_convolution(t,f,u,fker,fker_star,shape, solver, **solver_kwargs)

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    _, R_left = R_(t_left_new, f, u_hat)
    _, R_right = R_(t_right_new, f, u_hat)

    i = 0
    while (i < max_iter):
        print(t_left, t_right)
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            _, R_left = R_(t_left_new, f, u_hat)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            _, R_right = R_(t_right_new, f, u_hat)
        if (h < tol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(f,(1-t_opt)/t_opt, fker, fker_star, shape, **solver_kwargs) 

            return u, t_opt
        i += 1


def DP_func(A,v, lam_l, lam_r, solver, noise_std = 0.0, param = 1.0, u0 = None, **solver_kwargs):
    u = solver(A,v, lam_l,lam_r, u0 = u0, **solver_kwargs)
    return u, np.linalg.norm(np.dot(A,u) - v) - param * np.sqrt(np.prod(v.shape)) * noise_std

def DP_func_convolution(f, t, solver, fker, fker_star, shape, noise_std = 0.0, param = 1.0, **solver_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1

    if (t == 1):
        u = center(irfftn(rfftn(edge_pad_and_shift(f,m),shape)/fker),n,m)
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        u = solver(f, (1-t)/t, fker, fker_star, shape, **solver_kwargs)
    Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
    return u, np.linalg.norm(Au - f) - param * u.shape[0]* noise_std

def DP_secant(prob, v, solver, noise_std = 0.0, param = 1.0,
    Av = None,
    t_0 = 0.0, t_1 = 1.0, param_abstol = 1.0e-4, param_rtol = 1.0e-5, 
    param_max_iter = 100, **solver_kwargs):
    if Av is None:
        Av = np.dot(prob.A.T,  v) 
    

    # Calculate function values in the two starting values t_0 and t_1
    u = solver(prob,v,t_0,1.0-t_0, Av = Av, **solver_kwargs)
    DP_pp = np.linalg.norm(np.dot(prob.A,u) - v)**2 - param * np.prod(v.shape) * noise_std**2
    u = solver(prob,v,t_1,1.0-t_1, Av = Av, **solver_kwargs)
    DP_p = np.linalg.norm(np.dot(prob.A,u) - v)**2 - param * np.prod(v.shape) * noise_std**2
    t_pp = t_0
    t_p = t_1
    t_c_prev = t_1
    for i in range(param_max_iter):
        t_c = t_p - DP_p*(t_p - t_pp)/(DP_p - DP_pp)
        if (t_c < 0.0 or 1.0 < t_c):
            alpha = 1.0
            for m in range(1000):
                t_c = t_c_prev - alpha*DP_p*(t_p - t_pp)/(DP_p - DP_pp)
                if 0.0 <= t_c <= 1.0:
                    break
                alpha*=0.5
        u = solver(prob,v,t_c,1.0-t_c, Av = Av, u0 =u, **solver_kwargs)
        DP_c = np.linalg.norm(np.dot(prob.A,u) - v)**2 - param * np.prod(v.shape) * noise_std**2
        if np.abs(DP_c) < param_abstol:
            return u, t_c
        else:
            t_pp = t_p
            t_p = t_c
            DP_pp = DP_p
            DP_p = DP_c
            if (DP_p == DP_pp):
                return u, t_c
        t_c_prev = t_c
    return u,t_c

def DP_bisection(A, v, solver, noise_std = 0.0, param = 1.0,
    solve_method = 'eigh', decomposition = None, Av = None,
    t_left = 0.0, t_right = 1.0, param_abstol = 1.0e-4, param_rtol = 1.0e-5, 
    param_max_iter = 100, **solver_kwargs):

    if Av is None:
        Av = np.dot(A.T,  v) 
    
    if solve_method == 'eigh':
        if decomposition is None:
            decomposition = sp.linalg.eigh(np.matmul(A.T,A))
    elif solve_method == 'svd':
        if decomposition is None:
            decomposition = sp.linalg.svd(A)
    u = solver(A,v,t_left,1.0-t_left, solve_method = solve_method, decomposition = decomposition, Av = Av, **solver_kwargs)
    DP_left = np.linalg.norm(np.dot(A,u) - v)**2 - param * np.prod(v.shape) * noise_std**2
    for i in range(param_max_iter):
        t_mid = (t_left + t_right)/2
        if (t_right - t_left)/2 > param_rtol:
            u = solver(A,v,t_mid,1.0-t_mid, solve_method = solve_method, decomposition = decomposition, Av = Av, u0 = u, **solver_kwargs)
            DP_mid = np.linalg.norm(np.dot(A,u) - v)**2 - param * np.prod(v.shape) * noise_std**2
            if np.abs(DP_mid) < param_abstol:
                return u, t_mid
            else:
                if np.sign(DP_mid) == np.sign(DP_left):
                    t_left = t_mid
                    DP_left = DP_mid
                else:
                    t_right = t_mid
        else:
            return u, t_mid

def DP_secant_convolution(f, solver, fker, fker_star, shape, noise_std = 0.0, param = 1.1, t_0 = 1.0, t_1 = 0.99, DP_tol = 1.0e-8, **solver_kwargs):
    max_iter = 100
    u, DP_pp = DP_func_convolution(f, t_0, solver, fker, fker_star, shape, noise_std = noise_std, param = param, **solver_kwargs)
    u, DP_p = DP_func_convolution(f, t_1, solver, fker, fker_star, shape, noise_std = noise_std, param = param, **solver_kwargs)
    t_pp = t_0
    t_p = t_1
    for i in range(max_iter):
        t_c = t_p - DP_p*(t_p - t_pp)/(DP_p - DP_pp)
        u, DP_c = DP_func_convolution(f, t_c, solver, fker, fker_star, shape,noise_std = noise_std, param = param, **solver_kwargs)
        if np.abs(DP_c)/f.shape[0] < DP_tol:
            return u, t_c
        else:
            t_pp = t_p
            t_p = t_c
            DP_pp = DP_p
            DP_p = DP_c
    return u,t_c

def QOC(f, solver, lam_init = 1.0, q = 0.9, break_iter = 5, **solver_kwargs):
    max_iter = 100
    u_old = solver(f, lam_init, **solver_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = solver(f, lam, **solver_kwargs)
        loss = np.linalg.norm(u_new - u_old, 'fro')
        if(i == 1):
            minloss = loss
            loss_i = i
        else:
            if loss < minloss:
                minloss = loss
                loss_i = i
            else:
                if (i>= loss_i + break_iter):
                    break
        u_old = np.copy(u_new)
    lam = lam_init * (q ** loss_i)
    u = solver(f, lam, **solver_kwargs)

    return u, 1/(1 + lam)

def QOC_convolution(f, solver, fker, fker_star, shape, lam_init = 1.0, q = 0.9, break_iter = 5, **solver_kwargs):
    max_iter = 100
    u_old = solver(f, lam_init, fker, fker_star, shape, **solver_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = solver(f, lam,fker, fker_star, shape,  **solver_kwargs)
        loss = np.linalg.norm(u_new - u_old, 'fro')
        print(lam, loss)
        if(i == 1):
            minloss = loss
            loss_i = i
        else:
            if loss < minloss:
                minloss = loss
                loss_i = i
            else:
                if (i>= loss_i + break_iter):
                    break
        u_old = np.copy(u_new)
    lam = lam_init * (q ** loss_i)
    u = solver(f, lam,fker,fker_star,shape, **solver_kwargs)

    return u, 1/(1 + lam)


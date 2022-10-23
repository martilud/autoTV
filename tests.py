import autograd.numpy as np
from numpy.lib.arraysetops import union1d
from numpy.random.mtrand import beta
import scipy as sp
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from solvers import py_pdps_denoise, py_pdps_SR, py_quadratic_SR, py_quadratic_denoise, py_ridge, py_lasso_pg, py_lasso_pdhg, py_ridge_opt, py_pg_denoise, py_lasso_admm
from parameterselection import golden_section, opt, grid_search, DP_secant, DP_bisection,GCV, UPRE, LC
import imageio
from problem import *
from utils import add_gaussian_noise

def generatelineardata(N,M,M_nonzero, noise_std):
    betas = np.zeros(M)
    betas[:M_nonzero] = np.random.uniform(0.5,1.0,M_nonzero)
    signs = np.random.binomial(1,0.5, M)
    for i,sign in enumerate(signs):
        if sign:
            betas[i] = betas[i]
        else:
            betas[i] = -betas[i]

    # Data matrix X and response Y
    X = np.zeros((N,M)) 
    Y = np.zeros(N)

    noise_vec = np.random.normal(0.0,noise_std,N)
    # Covariance matrix of X
    cov_eigvecs = np.random.uniform(-1.0,1.0,M*M).reshape((M,M))
    cov = cov_eigvecs.T @ cov_eigvecs
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    corr = cov / outer_v
    corr[cov == 0] = 0
    chol = sp.linalg.cholesky(corr, lower=True) 
    X = np.random.normal(0, 1.0,M*N).reshape((N,M)) 
    X = np.dot(X,chol.T)
    Y = np.dot(X,betas) + noise_vec
    return X,Y, betas

def lassoapprox():
    np.random.seed(102)
    N = 5
    M = 2
    M_nonzero = 1
    noise_std = 0.1
    X,Y, u_true = generatelineardata(N,M,M_nonzero,noise_std)
    prob_svd = problem(A = X, solve_method = 'svd')
    resolution = 20
    xs = np.linspace(0.0,1.0, resolution)
    ts = np.linspace(0.0,1.0,100)
    us_ridge = np.zeros((100,M))
    #E = np.zeros(len(ts))
    #E_ridge = np.zeros(len(ts))
    #for i, t in enumerate(ts):
    #    print(i)
    #    us_ridge[i] = py_ridge(prob_svd,Y,t,1-t)
    #    u,_ = opt(prob_svd,Y,py_lasso_pg,us_ridge[i], tol = 1e-5)
    #    E[i] = np.linalg.norm(u - u_true)
    #    E_ridge[i] = np.linalg.norm(us_ridge[i] - u_true)
    #plt.plot(ts,E, label = "lasso")
    #plt.plot(ts,E_ridge, label = "ridge")
    #plt.plot(ts[np.argmin(E_ridge)], E_ridge[np.argmin(E_ridge)], 'x')
    #plt.legend()
    #plt.show()
    #E = np.zeros(len(xs))
    #flag = 0
    #for i,x in enumerate(xs):
    #    if x > u_true[0] and flag == 0:
    #        k = i
    #        flag = 1
    #    print(i)
    #    u_hat = np.zeros(M)
    #    u_hat[:M_nonzero] = x
    #    u,_ = opt(prob_svd,Y,py_lasso_pg,u_hat, tol = 1e-5)
    #    E[i] = np.linalg.norm(u - u_true)
    #plt.plot(xs,E)
    #for i in range(M_nonzero):
    #    plt.plot(u_true[i], E[k], 'x')
    #plt.show()
    #exit()
    ys = np.linspace(-1.0,1.0, resolution)
    xx,yy = np.meshgrid(xs,ys)
    Es = np.zeros((len(xs), len(xs)))
    for i,x in enumerate(xs):
        for j,y in enumerate(xs):
            print(i,j)
            u_hat = np.array([x,y])
            u,_ = opt(prob_svd, Y, py_lasso_pg,u_hat, tol=0.000001)
            Es[i,j] = np.linalg.norm(u - u_true)
    #print(Es)
    plt.imshow(Es)
    plt.show()
    print(u_true)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, Es)
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    plt.show()
def linearregressionexample():
    N = 50 # number of points
    M = 20 # Number of dimensions
    M_nonzero = 5 # Number of nonzero coefficients
    noise_std = 0.1
    N_simulations = 1
    np.random.seed(1)
    methods = ["Linear", "Opt Ridge", "GCV Ridge", "DP Ridge", 
            "LC Ridge", "Opt Lasso", "Lasso w. opt Ridge", "Lasso w. GCV Ridge", "Lasso w. DP Ridge", 
            "Lasso w. LC Ridge", "GCV Lasso", "DP Lasso", "UPRE Lasso", "LC Lasso"]
    us = np.zeros((len(methods), M))
    errors = np.zeros((N_simulations, len(methods)))
    for sim in range(N_simulations): 
        X,Y,u_true = generatelineardata(N,M, M_nonzero, noise_std)
        
        #OMPCV = OrthogonalMatchingPursuit(n_nonzero_coefs = M_nonzero, fit_intercept = False, normalize = False)
        #OMPCV = OrthogonalMatchingPursuitCV(fit_intercept = False, normalize = False, cv = 5, max_iter = M)
        #OMPCV.fit(X,Y)
        #u_omp = OMPCV.coef_
        #print(np.linalg.norm(u_omp - u_true))

        prob_eigh = problem(A = X,Aast =X.T, solve_method = 'eigh')
        prob_svd = problem(A = X,Aast =X.T, solve_method = 'svd')
        prob_svd.set_pseudoinverse()
        lam_l = 1.0
        lam_r = 1.0
        u_ridge,_ = py_ridge_opt(prob_svd,Y,1.0,1.0, u_hat = u_true)
        #t = time.time()
        #u_lasso,t_lasso = opt(prob_svd,Y,py_lasso_pdhg, u_ridge,loss = "projected", tol = 1e-5)
        u_opt,t_lasso = opt(prob_svd,Y,py_lasso_pdhg, u_true, loss = "projected",tol = 1e-5)
        #print(t_lasso)
        #t = time.time()
        u_lasso,t_lasso = grid_search(prob_svd,Y,py_lasso_pdhg, u_hat = u_ridge, loss = "alt", u_true = u_true,tol = 1e-5, plot = True)
        #print(t_lasso)
        #print("TIME", time.time() - t)
        t = time.time()
        #u_lasso2=  py_lasso_pdhg(prob_eigh, Y, 1.0,1.0,param_method = "opt",u_hat = u_true, tol = 1e-16,noise_std = noise_std, maxiter = 1000)
        print("TIME", time.time() - t)
        t = time.time()
        u_lasso3=  py_lasso_admm(prob_svd, Y, 1.0,1.0,param_method = "opt",u_hat = u_ridge, tau = 1,noise_std = noise_std, maxiter = 1000)
        print("TIME", time.time() - t)
        #print(np.linalg.norm(u_lassoomp - u_lasso3))
        print("OPT", np.linalg.norm(u_opt - u_true))
        print("RIDGE", np.linalg.norm(u_ridge - u_true))
        print("LASSORIDGE", np.linalg.norm(u_lasso - u_true))
        #print(np.linalg.norm(u_lassoomp - u_true))
        print("ADMM", np.linalg.norm(u_lasso3 - u_true))
        plt.plot(u_opt)
        plt.plot(u_lasso3)
        plt.plot(u_true)
        plt.show()
        exit()
        #print("t_ridge", t_ridge)
        u_lasso,t_lasso = opt(prob_svd,Y,py_lasso_pdhg,betas, loss = "projected")
        u_dp = py_lasso_pdhg(prob_svd,Y,1.0,1.0,param_method="dp",noise_std = sigma)
        u_our1,t_our = opt(prob_svd,Y,py_lasso_pdhg,u_ridge, loss = "projected")
        u_our2,t_our = opt(prob_svd,Y,py_lasso_pdhg,u_omp, loss = "projected")
        #print("t_lasso", t_lasso)
        #u_alt,t_alt = opt(prob_svd,Y,py_lasso_pdhg,u_ridge)#, loss = "projected")
        #print("t_alt", t_alt)
        #u,t = grid_search(prob_svd,Y,py_lasso_pdhg,u_ridge, u_true = betas, N = 200, loss = "projected", plot = True)
        print("RIDGE", np.linalg.norm(u_ridge - betas, 2))
        print("OMP", np.linalg.norm(u_omp - betas, 2))
        print("LASSO RIDGE", np.linalg.norm(u_our1 - betas, 2))
        print("LASSO OMP", np.linalg.norm(u_our2 - betas, 2))
        print("LASSO DP", np.linalg.norm(u_dp - betas, 2))
        print("LASSO OPT", np.linalg.norm(u_lasso - betas, 2))
        plt.plot(u_omp)
        plt.plot(u_our2)
        plt.plot(u_lasso)
        plt.plot(betas)
        plt.show()
        exit()
        us[0] = py_ridge(prob_eigh,Y,1.0,0.0)
        us[1],_ = opt(prob_eigh,Y,py_ridge,betas)
        us[2],_ = GCV(prob_eigh,Y,py_ridge)
        us[3],_ = DP_secant(prob_eigh,Y,py_ridge, noise_std=sigma)
        us[4],_ = LC(prob_eigh,Y,py_ridge)
        us[5],_ = opt(prob_svd,Y,py_lasso_pdhg,betas)
        us[6],_ = opt(prob_svd,Y,py_lasso_pdhg,us[1])
        us[7],_ = opt(prob_svd,Y,py_lasso_pdhg,us[2])
        us[8],_ = opt(prob_svd,Y,py_lasso_pdhg,us[3])
        us[9],_ = opt(prob_svd,Y,py_lasso_pdhg,us[4])
        us[10] = py_lasso_pdhg(prob_svd,Y,lam_l,lam_r, param_method = "GCV",noise_std=sigma, u_hat = betas) 
        us[11] = py_lasso_pdhg(prob_svd,Y,lam_l,lam_r, param_method = "dp",noise_std=sigma, u_hat = betas) 
        us[12] = py_lasso_pdhg(prob_svd,Y,lam_l,lam_r, param_method = "UPRE",noise_std=sigma, u_hat = betas) 
        us[13] = py_lasso_pdhg(prob_svd,Y,lam_l,lam_r, param_method = "LC",noise_std=sigma, u_hat = betas) 
        for m in range(len(methods)):
            errors[sim, m] = np.linalg.norm(us[m] - betas)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    ax.boxplot(errors, vert = 0)
    ax.set_yticklabels(methods)
    plt.title('M = ' + str(M) + ', N = ' + str(N) + ', sigma = ' + str(sigma) +  ', h = ' + str(M_nonzero/M) )
    plt.show()
    for i in range(len(methods)):
        print(methods[i] + f' {np.linalg.norm(us[i] - betas):.4f}')

def TVsuperresolution():
    u_true = imageio.imread('images/lenna.png')/255.0
    def A(u):
        return 0.25 * (u[::2,::2] + u[1::2,::2] + u[::2,1::2] + u[1::2,1::2])
    def Aast(v):
        out = np.zeros((v.shape[0] * 2, v.shape[1] * 2))
        out[::2,::2] = 0.25 * v
        out[1::2,::2] = out[::2,::2]
        out[::2,1::2] = out[::2,::2]
        out[1::2,1::2] = out[::2,::2]
        return out
    def AA(u):
        out = np.zeros_like(u)
        out[::2,::2] = 0.0625 * (u[::2,::2] + u[1::2,::2] + u[::2,1::2] + u[1::2,1::2])
        out[1::2,::2] = out[::2,::2]
        out[::2,1::2] = out[::2,::2]
        out[1::2,1::2] = out[::2,::2]
        return out
    
    sigma = 0.05
    v = add_gaussian_noise(A(u_true),sigma)
    n = 256
    new_n = n *2
    coords = np.zeros((2,n)) 
    coords[0] = np.arange(0,n,1); coords[1] = np.arange(0,n,1)
    coords_us = np.zeros((2,new_n))
    coords_us[0,:] = np.linspace(0.0, n - 1.0, new_n) 
    coords_us[1,:] = np.linspace(0.0, n - 1.0, new_n)
    print(coords_us[0].shape)
    f_us = sp.interpolate.interp2d(coords[0], coords[1], v, kind="linear")
    u_us = f_us(coords_us[0], coords_us[1])
    print(np.linalg.norm(u_us - u_true))
    prob_SR = problem(A = A, Aast = Aast, AA = AA, D = D_zero_boundary, Dast = Dast_zero_boundary, solve_method = "cg")
    prob_SR.set_pseudoinverse()
    u = py_pdps_SR(prob_SR,v,1.0,0.01,param_method="dp",u_hat = u_true,noise_std = sigma,maxiter = 100,verbose = False)
    #u_opt, _ = grid_search(prob_SR,v,py_quadratic_SR,u_hat = u_us, u_true = u_true, N = 100, plot = True)
    u_opt2, t = opt(prob_SR,v,py_quadratic_SR,u_hat = u_us, loss = "projected")
    u_opt1, t = opt(prob_SR,v,py_pdps_SR,u_hat = u_opt2, loss = "projected")
    #plt.imshow(u_opt)
    #plt.show()
    #print(np.linalg.norm(u - u_true))
    plt.imshow(u, cmap = "gray")
    plt.show()
    #plt.imshow(u_opt2, cmap = "gray")
    #plt.show()
    #plt.imshow(u_opt1, cmap = "gray")
    #plt.show()
    print(np.linalg.norm(u - u_true))
    #print(np.linalg.norm(u_opt - u_true))
    print(np.linalg.norm(u_opt2 - u_true))
    print(np.linalg.norm(u_opt1 - u_true))

def TVexample():
    u_true = imageio.imread('images/lenna.png')/255.0
    sigma = 0.1
    v = add_gaussian_noise(u_true,sigma)
    prob_denoise = problem(D = D_zero_boundary, Dast = Dast_zero_boundary, solve_method="identity")
    u_quad, _ = opt(prob_denoise,v,py_quadratic_denoise,u_true,tol = 1e-4)
    #u1 = py_pdps_denoise(prob_denoise,v,1.0,1.0, param_method = "opt", u_hat = u_true,maxiter = 200, verbose = False)
    t = time.time()
    #u3,_ = grid_search(prob_denoise,v, py_pg_denoise,u_hat = u_true,u_true = u_true, N = 50, plot = True)

    u1= py_pg_denoise(prob_denoise,v,1.0,1.0, param_method = "opt", u_hat = u_true, maxiter = 200, tol = 1e-10,verbose = True)
    #u1= py_pdps_denoise(prob_denoise,v,1.0,0.1, param_method = "dp", u_hat = u_quad, noise_std = sigma, maxiter = 100, tol = 1e-10,verbose = True)
    #u2,_ = py_pg_denoise(prob_denoise,v,lam,1.0, param_method = "op", u_hat = u_true, maxiter = 100, tol = 1e-10,verbose = False)
    print(time.time() - t)
    #u3,_ = opt(prob_denoise,v,py_pdps_denoise, u_true, tol = 1e-4)
    #t = time.time()
    #u3 = py_pdps_denoise(prob_denoise,v,0.01,0.01, param_method = "op", u_hat = u_true,verbose = True)
    #print(time.time() - t)

    #print(np.linalg.norm(u1 - u3))
    print(psnr(v,u_true))
    print(psnr(u1,u_true))
    print(psnr(u3,u_true))
    plt.imshow(u1)
    plt.show()
   
    exit()
    u_quad,_ = opt(prob_denoise, v, py_quadratic_denoise, u_true)
    plt.imshow(u_quad)
    plt.show()
    ts = np.linspace(0.5,0.9999,50)
    S = np.zeros(100)
    R = np.zeros(100)
    E = np.zeros(100)
    u = np.copy(u_quad)
    for i,t in enumerate(ts):
        print(i)
        u = py_pdps_denoise(prob_denoise,v,t,1-t,u0 = u, maxiter = 100)
        R[i] = np.linalg.norm(u - u_true)**2
        E[i] = np.linalg.norm(u - u_quad)**2
        S[i] = np.sum(u*(u_true - u_quad))
    plt.plot(S, color = "red")
    plt.plot(R, color = "blue")
    plt.plot(E, color = "green", linestyle ='dashed')
    plt.show()
   
    #u,t = grid_search(prob_denoise,v,py_pdps_denoise,u_hat = u_quad, u_true = u_true, N = 100, plot = True)
    exit()
    #u_quad,_ = opt(prob_denoise, v, py_quadratic_denoise, u_true)
    u_opt = py_pdps_denoise(prob_denoise,v,1.0,1.0, param_method = "opt",noise_std = sigma, u_hat = u_true, maxiter = 200, verbose = 1)
    u_opt_hat = py_pdps_denoise(prob_denoise,v,1.0,1.0, param_method = "opt",noise_std = sigma, u_hat = u_hat, maxiter = 200, verbose = 1)
    #u_opt_quad = py_pdps_denoise(prob_denoise,v,1.0,1.0, param_method = "opt",noise_std = sigma, u_hat = u_quad, maxiter = 200, verbose = 1)
    #u_dp = py_pdps_denoise(prob_denoise,v,1.0,1.0, param_method = "dp",noise_std = sigma, u_hat = u_true, maxiter = 200, verbose = 1)
    #u_dp = py_pdps_denoise(prob_denoise,v,1.0,0.01, param_method = "dp",noise_std = sigma, u_hat = u_true, maxiter = 1000, verbose = 1)
    print(np.linalg.norm(v - u_true))
    print(np.linalg.norm(u_opt - u_opt_hat))
    print(np.linalg.norm(u_opt - u_true))
    print(np.linalg.norm(u_opt_hat - u_true))

def treeexample():
    u_true = imageio.imread('images/lenna.png')/255.0
    #u_true = np.zeros((100,100))
    #u_true[20:80, 20:80] = 1

    # X Corresponds to the "x" and "y" values of a pixel
    pixels = np.arange(0,u_true.shape[0])
    XX,YY = np.meshgrid(pixels, pixels)
    XX = XX.reshape(np.prod(XX.shape))
    YY = YY.reshape(np.prod(YY.shape))
    X = np.zeros((2, len(XX)))
    X[0] = XX; X[1] = YY
    sigma = 0.05
    Y = add_gaussian_noise(u_true,sigma)
    print(np.linalg.norm(Y - u_true))
    Y = Y.reshape(np.prod(u_true.shape))
    tree = DecisionTreeRegressor(max_leaf_nodes = 15000)
    #etree = ExtraTreesRegressor(n_estimators=100)
    #rtree = RandomForestRegressor(criterion = "absolute_error", n_estimators=10, max_depth = 10)
    tree.fit(X.T,Y)
    y_tree = tree.predict(X.T)

    prob_denoise = problem(D = D_zero_boundary, Dast = Dast_zero_boundary, solve_method="identity")
    u_TV = py_pdps_denoise(prob_denoise,Y.reshape(u_true.shape), 1.0,0.1,param_method = "opt",u_hat= y_tree.reshape(u_true.shape), noise_std=sigma)
    u_TVopt = py_pdps_denoise(prob_denoise,Y.reshape(u_true.shape), 1.0,0.1,param_method = "opt",u_hat= u_true, noise_std=sigma)
    print(np.linalg.norm(u_true.reshape(np.prod(u_true.shape)) - y_tree))
    print(np.linalg.norm(u_true - u_TV))
    print(np.linalg.norm(u_true - u_TVopt))
    #plt.imshow(y_tree.reshape(u_true.shape), cmap = "gray")
    plt.imshow(Y.reshape(u_true.shape), cmap = "gray")
    plt.show()
    plt.imshow(y_tree.reshape(u_true.shape), cmap = "gray")
    plt.show()
    plt.imshow(u_TV, cmap = "gray")
    plt.show()
TVexample()

#lassoapprox()
#TVsuperresolution()
#linearregressionexample()
#treeexample()

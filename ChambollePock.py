from utils import *
from projectedGradient import *
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#def A(f,ker):
#    return fftpack.idctn(ff*fker, shape=s, norm='ortho') #signal.fftconvolve(f,ker,mode='same')

def ChambollePock_denoise_conv(f, lam, tau = 0.50, sig = 0.30, theta = 1.0, acc = False, tol = 1.0e-1, u0 = None):
    if u0 is  None:
        #u = np.copy(f)
        #u_prev = np.copy(f)
        #u_hat = np.copy(f)
        u = np.zeros(f.shape, f.dtype)
        u_prev = np.zeros(f.shape, f.dtype) 
        u_hat = np.zeros(f.shape, f.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_hat = np.copy(u0)

    #p_hat = sig*grad(u_hat)
    #p = lam * p_hat / np.maximum(lam, better_norm1(p_hat))
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    maxiter = 500
    dualitygap_list= np.zeros(maxiter+1)
    dualitygap_list[0] = dualitygap_denoise(lam,u,divp,f)
    psnr_list = np.zeros(maxiter)
    #psnr_list[0] = psnr(u,f)
    if acc:
        gam = 0.5
    i = 0
    res= tol + 1.0
    plot_iterations = [10, 100, 500]
    while (i < maxiter and res > tol): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam, better_norm1(p_hat))
        divp = div(p)
        u = 1/(1 + tau) * (u + tau * divp + tau *f) 
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
        dualitygap_list[i+1] = dualitygap_denoise(lam,u,divp,f)
        res = dualitygap_list[i+1]/dualitygap_list[0]
        if (i+1 in plot_iterations):
            plt.imsave("Lenna" + str(i) + ".png", u, cmap = "gray")
        psnr_list[i] = psnr(u,u_prev)
        i+=1
        #print(i)
    return u, dualitygap_list, psnr_list, i 
 
def ChambollePock_denoise(f, lam, tau = 0.50, sig = 0.30, theta = 1.0, acc = False, tol = 1.0e-1, u0 = None):

    if u0 is  None:
        #u = np.copy(f)
        #u_prev = np.copy(f)
        #u_hat = np.copy(f)
        u = np.zeros(f.shape, f.dtype)
        u_prev = np.zeros(f.shape, f.dtype) 
        u_hat = np.zeros(f.shape, f.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_hat = np.copy(u0)
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    maxiter = 100
    dualitygap_init = dualitygap_denoise(lam,u,divp,f)
    if acc:
        gam = 0.5
    i = 0
    res= tol + 1.0
    #plot_iterations = [10, 100, 500]
    while (i < maxiter and res > tol): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam,norm1(p_hat))
        divp = div(p)
        u = 1/(1 + tau) * (u + tau * divp + tau *f) 
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
        if (i%10 == 0):
            res = dualitygap_denoise(lam,u,divp,f)/dualitygap_init
        #if (i+1 in plot_iterations):
        #    plt.imsave("Lenna" + str(i) + ".png", u, cmap = "gray")
        i+=1
        #print(i)
    return u   
if __name__ == "__main__":
    Originalf = imageio.imread('images/Lenna.jpg', pilmode = 'F')
    Originalf = Originalf/255.0
    f = np.copy(Originalf)
    shape = f.shape
    nx = f.shape[0]
    ny = f.shape[1]
    sigma = 0.1
    f = add_gaussian_noise(f,sigma)

    lam = 0.1
    #result,costlist = ChambollePock_matrix(f, lam, A, tau = 0.10, sig = 0.10, theta = 1.0)
    result1,costlist1 = ChambollePock_denoise(f,lam,tau=0.25, sig = 0.25,theta = 1.0, tol = 1.0e-2)
    #result1, costlist1 = projected_gradient(f,lam,tau = 0.25, tol = 1.0e-10)
    #plt.semilogy(costlist, '-')
    plt.loglog(costlist1)
    plt.grid()
    plt.show()
    plt.imshow(result1, cmap='gray')
    plt.show()
    ##ker = np.outer(signal.gaussian(2*nx,sigma), signal.gaussian(2*ny, sigma))
    ##ker = ker[nx:, ny:]
    ##ker = ker/np.sum(ker)
    ##print(ker)
    ##plt.imshow(ker,cmap='gray')
    ##plt.show()
    ##s = [n + k - 1 for n,k in zip(f.shape, ker.shape)]
    ##s = np.array(f.shape) #+ np.array(ker.shape) - 1
    ##
    ##fker = fftpack.dctn(ker,shape=s, norm='ortho')
    ##fker2 = np.abs(fker)**2
    ##
    ##fker_conj = fftpack.dctn(ker[::-1],shape=s, norm='ortho')
    ##
    ##ff = fftpack.dctn(f, shape=s, norm='ortho')
    ##f = A(f,ker)
    ##print(f)
    ##plt.imshow(f,cmap='gray')
    ##plt.show()
    #
    fig = plt.figure()
    ax = fig.subplots(1,2)
    ax[0].imshow(f, cmap = 'gray')
    ax[1].imshow(result1, cmap = 'gray')
    plt.show()


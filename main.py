import numpy as np
from PIL import Image
import imageio
from utils import *
from chambollepock import py_cp_denoise, ChambollePock_denoise, py_cp_denoise_dp, py_pg_denoise_dp

def es_denoise(tau):
        #f = add_gaussian_noise(u,sigma)
        u = imageio.imread('images/lenna.png')/255.0
        sigma = 0.1
        f = add_gaussian_noise(u,sigma)
        u,t = py_cp_denoise_dp(f,0.1,u,u0 = f,noise_sig=sigma, tau = tau)
        plt.imshow(u)
        plt.show()

if __name__ == '__main__':  
    np.random.seed(318)
    u = imageio.imread('images/lenna.png')/255.0
    sigma = 0.05
    f = add_gaussian_noise(u,sigma)
    t0 = time.time()
    u = py_cp_denoise(f,16.0,u0=f)
    print(time.time() - t0)
    t0 = time.time()
    u,t = py_cp_denoise_dp(f,0.1,u,u0 = f,noise_sig=sigma,tau = 1.0, sig = 1.0/16.0)
    print(t)
    print(time.time() - t0)
    t0 = time.time()
    u,t = py_pg_denoise_dp(f,0.1,u,u0 = f,noise_sig=sigma, tau = 0.25)
    print(t)
    print(time.time() - t0)
    #res1 = ChambollePock_denoise(f,0.1)
    #res,t = py_cp_denoise_dp(f, 0.99, u, noise_sig = sigma, u0 = f, tau = 1.0, sig = 1.0/16.0, theta = 1.0)
        #tau_list = np.linspace(0.01,1.0,10)
    #iter_list = np.zeros(10)
    #for j in range(10):
    #    print(j)
    #    iter_list[j] = es_denoise(tau_list[j])
    #plt.plot(tau_list,iter_list)
    #plt.show()
    #exit()
    #es = EvolutionStrategy(np.array([0.25]),es_denoise, population_size=4, sigma=0.1, learning_rate=0.01, decay=0.995, num_threads=-1)
    #es.run(10, print_step=1)
    #print(es.get_weights())

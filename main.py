from utils import *
from ChambollePock import *
from projectedGradient import *
import time
from PIL import Image
import os
import numpy.random
from comparison_methodsTV import *
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim
#from interpolation import *
from skimage.transform import resize


def createDownsampled_old(f,n):
    shape = f.shape
    if (not (shape[0]/n).is_integer()):
        print("CHOOSE NUMBER SMART")
    m = shape[0]//n
    image_array = np.zeros((n,m,m))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                image_array[i,j,k] = f[i + j*n, i + k*n]
    return image_array

def createDownsampled(u,n,blur = False, sigma_blur = 0.5):
    shape = u.shape
    if blur:
        f = scipy.ndimage.gaussian_filter(u, sigma = sigma_blur)
    else:
        f = np.copy(u)

    l = int(np.sqrt(n))
    m = shape[0]//n * l
    image_array = np.zeros((n,m,m))
    shift_x = 0
    shift_y = 0
    for i in range(n):
        for j in range(m):
            for k in range(m):
                image_array[i,j,k] = f[shift_x + j*l, shift_y + k*l]
        if ((shift_x+1)%l == 0 and shift_x != 0):
            shift_x =0
            shift_y +=1
        else:
            shift_x += 1
    return image_array

def createEstimator_recursive(f, n,depth, curr_depth, blur = False, sigma_blur = 0.5):
    """
    Choose n = 4
    """
    if curr_depth == 0:
        image_array = createDownsampled(f,n, blur = blur, sigma_blur = sigma_blur)
        image_array_upsampled = createUpsampled(image_array, f.shape)
        #fig = plt.figure(figsize=(2,2))
        #for i in range(n):
        #    fig.add_subplot(2,2,i+1)
        #    plt.imshow(image_array_upsampled[i], cmap = "gray")
        #plt.show()

        return np.average(image_array_upsampled, axis=0)
    else:
        image_array = createDownsampled(f,n)
        for i in range(n):
            image_array[i] = createEstimator_recursive(image_array[i], n, depth, curr_depth-1, blur = blur, sigma_blur = sigma_blur)
        image_array_upsampled = createUpsampled(image_array, f.shape)
        return np.average(image_array_upsampled, axis=0)

def createDownsampled_random(u,n,N,blur = False, sigma_blur = 0.5):
    shape = u.shape
    if blur:
        f = scipy.ndimage.gaussian_filter(u, sigma = sigma_blur)
    else:
        f = np.copy(u)
    l = int(np.sqrt(n))
    m = shape[0]//n * l
    newshape = (m,m)
    image_array = np.zeros((N,m,m))
    cord_array = np.arange(0,l)
    for i in range(N):
        x_cords = np.random.choice(cord_array, m*m)
        y_cords = np.random.choice(cord_array, m*m)
        for j in range(m):
            for k in range(m):
                image_array[i,j,k] = f[x_cords[j*m + k] + j*l,y_cords[j*m + k] + k*l]
    return image_array

def createUpsampled(image_array, orig_shape):
    shape = image_array.shape
    image_array_upsampled = np.zeros((shape[0], ) + orig_shape)
    for i in range(shape[0]):
        #image = Image.fromarray(scipy.ndimage.gaussian_filter(image_array[i], sigma = 0.6))
        image = Image.fromarray(image_array[i])
        image = image.resize(orig_shape, Image.BILINEAR)
        image_array_upsampled[i] = np.array(image)
    return image_array_upsampled

def gram_schmidt_vec(image_array):
    shape = image_array.shape
    image_array_vec = image_array.reshape((shape[0], shape[1]*shape[2]))
    basis_img = np.zeros((shape[0],shape[1]*shape[2]))
    next_img = image_array_vec[0]/np.linalg.norm(image_array[0])
    basis_img[0] = next_img
    for i in range(1, shape[0]):
        proj_sum = np.zeros((shape[1]*shape[2]))
        next_img = image_array_vec[i]
        for j in range(i):
            proj_sum += np.inner(next_img, basis_img[j]) * next_img
        next_img = next_img - proj_sum 
        basis_img[i] = next_img/np.linalg.norm(next_img)
    return basis_img

def gram_schmidt(image_array):
    shape = image_array.shape
    basis_img = np.zeros(shape)
    next_img = image_array[0]/np.linalg.norm(image_array[0], 'fro')
    basis_img[0] = next_img
    for i in range(1, shape[0]):
        proj_sum = np.zeros((shape[1],shape[2]))
        next_img = image_array[i]
        for j in range(i):
            proj_sum += np.linalg.norm(np.multiply(next_img, basis_img[j]), 'fro') * basis_img[j]
        next_img = next_img - proj_sum 
        basis_img[i] = next_img/np.linalg.norm(next_img, 'fro')
    return basis_img

def modified_gram_schmidt(image_array):
    shape = image_array.shape
    basis_img = np.zeros(shape)
    basis_img_curr = np.zeros((shape[1],shape[2]))
    for i in range(shape[0]):
        basis_img_curr = image_array[i]
        for j in range(i):
            basis_img_curr = basis_img_curr - np.sum(np.multiply(basis_img[j], basis_img_curr)) * basis_img[j]
        basis_img[i] = basis_img_curr/np.linalg.norm(basis_img_curr, 'fro')
    return basis_img

def modified_gram_schmidt_vec(image_array):
    shape = image_array.shape
    image_array_vec = image_array.reshape((shape[0], shape[1]*shape[2]))
    basis_img = np.zeros((shape[0],shape[1]*shape[2]))
    basis_img_curr = np.zeros(shape[1]*shape[2])
    for i in range(shape[0]):
        basis_img_curr = image_array_vec[i]
        for j in range(i):
            basis_img_curr = basis_img_curr - np.inner(basis_img[j], basis_img_curr)* basis_img[j]
        basis_img[i] = basis_img_curr/np.linalg.norm(basis_img_curr)
    return basis_img

def testMean():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    sigma = 0.1
    n = 1000
    image_array = np.zeros((n,)+u.shape)
    avg_image_array = np.zeros((n,)+u.shape)
    psnr_array = np.zeros(n)
    avg_img = np.zeros(u.shape)
    for i in range(n):
        image_array[i] = add_gaussian_noise(u,sigma)
        avg_image_array[i] =np.average(image_array[:(i+1)], axis= 0)
        psnr_array[i] = psnr(u,avg_image_array[i])
    plt.plot(psnr_array)
    plt.show()


def testDownsampled():
    u =  imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sigma = 0.02
    f = add_gaussian_noise(u,sigma)

    orig_shape = f.shape
    n = 4
    n_rec = 4
    nn = np.ceil(np.sqrt(n))

    sigma_blur = 0.1

    avg_img_recursive = createEstimator_recursive(f,n = n_rec, depth = 1, curr_depth =1, blur = True, sigma_blur = sigma_blur/4)
    image_array_random = createDownsampled_random(f,n, N = 10, blur = True, sigma_blur = sigma_blur)
    image_array_fixed = createDownsampled(f,n, blur = True, sigma_blur = sigma_blur)
    image_array_upsampled_random = createUpsampled(image_array_random,orig_shape)
    image_array_upsampled_fixed = createUpsampled(image_array_fixed,orig_shape)
    #fig = plt.figure(figsize=(nn,nn))
    #for i in range(n):
    #    fig.add_subplot(nn,nn,i+1)
    #    plt.imshow(image_array_random[i], cmap = "gray")
    #plt.show()

    avg_img_random = np.average(image_array_upsampled_random, axis=0)
    avg_img_fixed = np.average(image_array_upsampled_fixed, axis=0)

    num_plot = 4
    fig,ax = plt.subplots(1, num_plot)
    ax[0].imshow(f, cmap="gray")
    ax[1].imshow(avg_img_random, cmap = "gray")
    ax[2].imshow(avg_img_fixed, cmap = "gray")
    ax[3].imshow(avg_img_recursive, cmap = "gray")
    plt.show()

    print("SSIM orig:", ssim(f,u, data_range = 1.0))
    print("PSNR orig:", psnr(f,u))
    print("SSIM random:", ssim(f,avg_img_random, data_range = 1.0))
    print("PSNR random:", psnr(avg_img_random,u))
    print("SSIM fixed:", ssim(f,avg_img_fixed, data_range = 1.0))
    print("PSNR fixed:", psnr(avg_img_fixed,u))
    print("SSIM recursive:", ssim(u,avg_img_recursive, data_range = 1.0))
    print("PSNR recursive:", psnr(u,avg_img_recursive))

    #t_opt, R_list = gridSearch(f, u, 50, plot=False)
    #t_opt_hat, R_list_hat = gridSearch(f, avg_img, 50, plot=False)
    #plt.plot(R_list)
    #plt.plot(R_list_hat)
    #plt.show()
    #u_opt_DC, t_opt_DC = discrepancy_ruleTV(f, sigma, tau = 1.0, lam_init = 1.0, q = 0.9)
    tol = 1e-4
    u_opt, t_opt = optTV_golden(f,u, tol = tol)
    print("SSIM rec orig:", ssim(f,u_opt, data_range = 1.0))
    print("PSNR rec orig:",psnr(u_opt,u))

    u_opt_recursive, t_opt_recursive = optTV_golden(f, avg_img_recursive, tol = tol)
    print("SSIM rec recursive:", ssim(f,u_opt_recursive, data_range = 1.0))
    print("PSNR rec recursive:", psnr(u_opt_recursive,u))

    u_opt_random, t_opt_random = optTV_golden(f, avg_img_random, tol = tol)
    print("SSIM rec random:", ssim(f,u_opt_random, data_range = 1.0))
    print("PSNR rec random:", psnr(u_opt_random,u))

    u_opt_fixed, t_opt_fixed = optTV_golden(f, avg_img_fixed, tol = tol)
    print("SSIM rec fixed:", ssim(f,u_opt_fixed, data_range = 1.0))
    print("PSNR rec fixed", psnr(u_opt_fixed,u))

    print("t opt, random, fixed, recursive:", t_opt, t_opt_random, t_opt_fixed, t_opt_recursive)

    fig,ax = plt.subplots(1, num_plot)
    ax[0].imshow(u_opt, cmap="gray")
    ax[1].imshow(u_opt_random, cmap = "gray")
    ax[2].imshow(u_opt_fixed, cmap = "gray")
    ax[3].imshow(u_opt_recursive, cmap = "gray")
    plt.show()

    #print(t_opt, t_opt_hat, t_opt_DC )
    #print(psnr(u_opt_DC,u))
    exit()
    """
    image_array_upsampled = createUpsampled(image_array,orig_shape)
    fig = plt.figure(figsize=(nn,nn))
    for i in range(n):
        fig.add_subplot(nn,nn,i+1)
        plt.imshow(image_array_upsampled[i], cmap = "gray")
    plt.show()
    #basis_img = modified_gram_schmidt_vec(image_array_upsampled)
    """

    t_opt_hat, R_list = gridSearch(f, avg_img, N = 30, plot=True) 
    basis_img = modified_gram_schmidt(avg_img - image_array_upsampled[:-1])

    #basis_img_img = basis_img.reshape((basis_img.shape[0],) +  orig_shape)
    fig = plt.figure(figsize=(nn,nn))
    for i in range(n-1):
        fig.add_subplot(nn,nn,i+1)
        #print(np.inner(basis_img[0], basis_img[i]))
        #print(np.linalg.norm(basis_img[i]))
        plt.imshow(basis_img[i], cmap = "gray")
        #plt.imshow(basis_img_img[i], cmap = "gray")
    plt.show()
    h = 8
    basis_img = basis_img[:h, :]
    N = 50
    #t_opt = 0.9228 #256 image
    #t_opt, R_list = gridSearch(f,u,N)
    t_opt_hat, t_list,  R_list_hat = gridSearch_basis(f, basis_img, avg_img, N)
    #print(t_opt, t_opt_hat)
    plt.plot(t_list, R_list_hat)
    plt.vlines(t_opt, ymin = np.min(R_list_hat), ymax = np.max(R_list_hat))
    #plt.plot(R_list)
    plt.show()

def testInterpolation():
    u = imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sig = 0.01
    f = add_gaussian_noise(u, sigma = sig)
    print(psnr(u,f))
    newshape = (256,256)
    shape = u.shape
    sigma_blur = 1.2
    image_array_fixed = createDownsampled(f,n=4, blur = True, sigma_blur = sigma_blur)
    image_array_upsampled_fixed = createUpsampled(image_array_fixed,shape)
    avg_img = np.average(image_array_upsampled_fixed, axis = 0)
    print(psnr(avg_img,u))
    #xx,yy = np.meshgrid(np.arange(u.shape[0]),np.arange(u.shape[1]))
    #print(xx.shape)
    #plt.scatter(xx,yy, c=u[::-1],s=0.01)
    #plt.show()
    order_down = 0
    order_up = 5
    aa = True
    aa_up = False
    aa_sig = 1.0
    sigmas = np.linspace(0,2,10)
    modes = ['constant', 'edge', 'symmetric','reflect','wrap']
    mode = 'symmetric'
    result_mat = np.zeros((6,6))
    max_psnr = 0.0
    for sig in sigmas:
        print(sig)
        gauss = resize(f, shape, order = 0, mode = mode, anti_aliasing = aa, anti_aliasing_sigma = sig)
        print(psnr(gauss,u))
        for orderd in range(6):
            for orderu in range(6):
                down = resize(f, newshape, order = orderd, mode=mode, anti_aliasing = aa,anti_aliasing_sigma = sig)
                up = resize(down, shape, order = orderu, mode=mode, anti_aliasing = aa_up)
                result_mat[orderd, orderu] = psnr(up,u)
                #print(orderd, orderu, psnr(up,u))
        if(np.max(result_mat)> max_psnr):
            max_psnr = np.max(result_mat)
            max_list = [max_psnr, np.where(result_mat == np.max(result_mat)), sig, mode]

        print(result_mat)
        print(np.max(result_mat))
        print(np.where(result_mat == np.max(result_mat)))
    print(max_list)
    #result1 = nearest(u, newshape)
    #image = Image.fromarray(u)
    #image = image.resize(newshape, Image.NEAREST)
    #result2 = np.array(image)
    #print(psnr(result1,result2))
    #plt.imshow(result1,cmap = "gray")
    #plt.show()

def test_gauss():
    u = imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sig = 0.1
    f = add_gaussian_noise(u, sigma = sig)
    #plt.imshow(f, cmap= 'gray')
    #plt.show()
    print(psnr(u,f))
    blur_sigs = np.linspace(0,3,100)
    blurs = np.zeros(100)
    i = 0
    for blur_sig in blur_sigs:
        blurred = scipy.ndimage.gaussian_filter(f,blur_sig)
        blurs[i] = psnr(blurred,u)
        i+=1
    best_blur_sig = blur_sigs[np.argmax(blurs)]
    blurred = scipy.ndimage.gaussian_filter(f,best_blur_sig/2)
    print(psnr(blurred,u))
    shape = f.shape
    image_array_fixed = createDownsampled(f,n=4, blur = True, sigma_blur = 0.0)
    image_array_upsampled_fixed = createUpsampled(image_array_fixed,shape)
    avg_img = np.average(image_array_upsampled_fixed, axis = 0)
    print(psnr(avg_img,u))

    tol = 1e-4
    u_opt, t_opt = optTV_golden(f,u, tol = tol)
    print("SSIM rec orig:", ssim(f,u_opt, data_range = 1.0))
    print("PSNR rec orig:",psnr(u_opt,u))

    u_opt_blurred, t_opt_recursive = optTV_golden(f, blurred, tol = tol)
    print("SSIM rec recursive:", ssim(f,u_opt_blurred, data_range = 1.0))
    print("PSNR rec recursive:", psnr(u_opt_blurred,u))
    u_opt_fixed, t_opt_fixed = optTV_golden(f, avg_img, tol = tol)
    print("PSNR rec recursive:", psnr(u_opt_fixed,u))

def test_gauss_opt_sig():
    u = imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    sig = 0.08
    f = add_gaussian_noise(u, sigma = sig)
    N = 100
    blur_sigs = np.linspace(0,3,N)
    u_hats = np.zeros((N,)+ f.shape)
    u_ts = np.zeros((N,)+ f.shape)
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    for i in range(1,N-1):
        print(i)
        t = t_list[i]
        u_ts[i] = ChambollePock_denoise(f, (1-t)/t, tau = 0.4, sig = 0.25, acc = True, tol = 1e-4) 
        u_hats[i] = scipy.ndimage.gaussian_filter(f,blur_sigs[i])
    Rs = np.zeros((N,N))
    for i in range(1,N-1):
        for j in range(1,N-1):
            print(i,j)
            Rs[i,j] = np.linalg.norm(u_ts[i] - u_hats[j])**2

    tt, ss = np.meshgrid(t_list, blur_sigs)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(tt, ss, Rs,
                       linewidth=0, antialiased=False)
    plt.show()

if __name__ == "__main__":
    np.random.seed(316)
    #test_gauss_opt_sig()
    test_gauss()
    #testInterpolation()
    #testMean()
    #testDownsampled()
    #TV_convergence()
    #testPokemon()
    #testOptTV()
    #testParameterSelections()
    #testEstimator()
    #exit()
    #TV_convergence()
    #testOtherParameterSelections()
    exit()    
    n = 32
    nn = n*n
    h = 128
    sigma = 0.1
    N_train = 500
    #train = import_train(N_train, n)
    #train, test = import_cifar()
    image_array = better_generate_test_images(512)
    plt.imsave('Lenna_grayscale.png',image_array[0,:,:], cmap= 'gray')
    rand = random.randint(0,N_train)
    test = train[rand,:,:]
    train = np.delete(train, rand, 0).reshape(N_train - 1, n,n)
    train_shape = train.shape
    test_shape = test.shape
    train = train + sigma * np.random.normal(0,1,np.prod(train_shape)).reshape(train_shape) 
    test_noisy = test + sigma * np.random.normal(0,1,np.prod(test_shape)).reshape(test_shape) 
    
    Pi = pi_hat_n(covariance_matrix(train), h = h)
    X_hat = np.dot(test_noisy.reshape(nn), Pi).reshape(n,n)
    t_opt, R_list = gridSearch(test_noisy[:,:],test[:,:], False)
    t_opthat, R_listhat = gridSearch(test_noisy[:,:],X_hat[:,:], False)
    t_list = np.linspace(0,1, len(R_list))
    plt.plot(t_list, R_list, label="True")
    plt.plot(t_list, R_listhat, label = "Empirical")
    plt.xlabel('t')
    plt.ylabel('R')
    plt.grid()
    plt.legend()
    plt.show()
    t_optTV = optTV(test_noisy[:,:], X_hat)
    result, _ = ChambollePock_denoise(test[:,:], (1-t_optTV)/t_optTV)

    #N_train = 100
    #N_test = 1
    #sigma = 0.1
    #image_array = better_generate_test_images(n)
    #amount = [0,10]
    #Pi, X_test, Y_test = set_it_up_image(image_array, amount, (n,n), h, N_train, N_test, sigma = sigma)
    ##piY_n, X_test, Y_test = set_it_up_id(nn, h, N_train, N_test, sigma)
    #
    ##X_test = np.zeros((n,n))
    ##X_test[24:75,24:75] = 1.0
    ##Pi, X_test, Y_test = set_it_up_id(nn,h,N_train,N_test,sigma)
    #X_hat = np.dot(Y_test.reshape(nn), Pi).reshape(n,n)
    #test_image = Image.fromarray(Y_test[0,:,:]*255.0)
    ##test_image = test_image.resize((256,256))
    #
    ##X_test = X_test.reshape((n,n))#X_test.reshape((N_test,n,n))
    ##Y_test = Y_test.reshape((n,n))#Y_test.reshape((N_test,n,n))
    ###X_hat = X_hat.reshape((n,n))
    ##plt.plot(Y_test.T)
    ##plt.plot(X_hat.T)
    ##plt.show()
    #
    fig = plt.figure()
    ax = fig.subplots(1,3)
    ax[0].imshow(test_noisy, cmap = 'gray')
    ax[0].set_title("X_test")
    ax[1].imshow(X_hat, cmap ='gray')
    ax[1].set_title("Y_test")
    ax[2].imshow(result, cmap = 'gray')
    ax[2].set_title("Result")
    plt.show()
    #
    #t_opt, R_list = gridSearch(Y_test[0,:,:],X_test[0,:,:], False)
    #t_opthat, R_listhat = gridSearch(Y_test[0,:,:],X_hat[:,:], False)
    #

    #
    #t = time.time()
    #t_optTV = optTV(Y_test[0,:,:], X_hat)
    #print(t_opt, t_opthat, t_optTV)
    #print(time.time() - t)
    #result, _ = ChambollePock_denoise(Y_test[0,:,:], (1-t_opthat)/t_opthat)
    #
    #
    #fig = plt.figure()
    #ax = fig.subplots(1,3)
    #ax[0].imshow(Y_test[0,:,:], cmap ='gray')
    #ax[0].set_title("Noisy image")
    #ax[1].imshow(X_hat[:,:], cmap = 'gray')
    #ax[1].set_title("Empirical estimator")
    #ax[2].imshow(result, cmap = 'gray')
    #ax[2].set_title("OptTV result")
    #plt.show()
    #
    ##t_optalg = backtracking_interpolate(Y_test, X_hat, tol = 1e-6)
    #
    ##print(time.time() - t)
    ##methods = ['discrepancy_ruleTV', 'monotone_error_ruleTV']
    ##for method in methods:
    ##    t = time.time()
    ##    z, lam = eval(method)(Y_test,sigma)
    ##    print(method, "lambda: ", lam, "time: ", time.time()-t)
    ##
    ##plt.plot(X_hat, label="X HATTE")
    ##plt.plot(X_test.T, label="X TRU")
    ##plt.legend()
    ##plt.show()
    #
    #
    ##plt.plot(result, label='result')
    ##plt.plot(X_train[0,:], label= 'not result')
    ##plt.legend()
    ##plt.show()
    Lenna  = imageio.imread('images/lenna.ppm', pilmode = 'F')/255.0


from utils import *
from ChambollePock import *
from projectedGradient import *
import time
from PIL import Image
import os
import random
from resizeimage import resizeimage
from comparison_methodsTV import *
import scipy.ndimage


def generate_test_images(n):
    image_array = np.zeros((2,n,n))
    image_array[0,n//8:n*5//8,n//4:n*3//4] = 1.0
    image_array[1,n*3//8:n*7//8,n//4:n*3//4] = 1.0
    return image_array

def generate_Lenna_test_images(n):
    image_array = np.zeros((1, n,n))
    Lenna  = Image.open('images/lenna256.jpg')#/255.0
    #Cameraman  = Image.open('images/cameraman256.png')#/255.0
    Lenna = Lenna.resize((n,n), Image.BICUBIC)
    #Cameraman = Cameraman.resize((n,n), Image.BICUBIC)
    image_array[0,:,:] = np.dot(np.array(Lenna)[...,:3], [0.2989, 0.5870, 0.1140])/255.0
    #image_array[1,:,:] = np.array(Cameraman).reshape(n,n)/255.0
    return image_array

def import_cifar():
    import pickle
    with open('images/cifar-100/train', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    train = dict[b'data']

    with open('images/cifar-100/test', 'rb') as fi:
        dict = pickle.load(fi, encoding='bytes')
    test = dict[b'data']

    train = train.reshape((50000,32,32,3), order = 'F')
    test = test.reshape((10000,32,32,3), order = 'F')
    return np.dot(train[...,:3], [0.2989, 0.5870, 0.1140])/255.0, np.dot(test[...,:3], [0.2989, 0.5870, 0.1140])/255.0

def import_train(N = 721,n = 120):
    image_array = np.zeros((N, n, n))
    for i,filename in enumerate(os.listdir('images/pokemon-better')):
        image_array[i] = imageio.imread('images/pokemon-better/'+filename, pilmode = 'F')/255.0
        #image = image.resize((n,n), Image.BICUBIC)
        #m_arr = np.fromstring(image.tobytes(), dtype=np.float32)
        #image = np.array(image.getdata())/255.0
        #image = image.reshape((n, n), order = 'C')  
        #plt.imshow(image, cmap = "gray")
        #plt.show()
    return image_array

def plot3d(f):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    shape = f.shape
    nx = f.shape[0]
    ny = f.shape[1]
    X = np.arange(0, 1, 1.0/nx)
    Y = np.arange(0, 1, 1.0/ny)
    X, Y = np.meshgrid(X, Y)

# Plot the surface.
    surf = ax.plot_surface(X, Y, f,cmap = 'gray')
    ax.view_init(70,105)
    plt.show()

def TV_convergence():
    u = imageio.imread('images/lenna256.jpg', pilmode = 'F')
    u = u/255
    f = np.copy(u)
    sigma = 0.1
    f = add_gaussian_noise(f,sigma)
    print("psnr:", psnr(f,u))
    print("TV:", TV(f))
    t = 0.9
    lam = (1-t)/t
    print("lam", lam)
    par = TV(f)*lam
    figstring = "High.png"
    print("lam*TV(f)", str(round(par,2)))

    ctau_list = [0.50, 0.25]
    csig_list = [0.50, 0.25]
    clinestyles = ['solid', 'dotted', 'dashed','dashdot']
    i = 0

    #for tau in ctau_list:
    #    for sig in csig_list:
    #        t = time.time()
    #        _, list, _, iter = ChambollePock_denoise_conv(f,lam, tau = tau, sig = sig, tol = 0.0)
    #        print("tau: ", tau, "sig", sig, "time", time.time() - t)
    #        print("duality", list[iter])
    #        plt.semilogy(list, label = r"$\tau =$ " + str(tau) + ", " + r"$\sigma =$ " + str(sig), linestyle = clinestyles[i])
    #        i+=1
    #plt.xlabel("Iterations")
    #plt.ylabel("Duality gap")
    #plt.title("ChambollePock, " + r"$\lambda TV(f) = $" + str(round(par,2)))
    #plt.legend()
    #plt.savefig("Chambolle_" + figstring, format="png", bbox_inches="tight")
    #plt.clf()
    i = 0
    for tau in ctau_list:
        for sig in csig_list:
            t = time.time()
            _, list, _, iter = ChambollePock_denoise_conv(f,lam, tau = tau, sig = sig, acc = True, tol = 0.0)
            print("tau: ", tau, "sig", sig, "time", time.time() - t)
            print("duality", list[iter])
            plt.semilogy(list, label = r"$\tau =$ " + str(tau) + ", " + r"$\sigma =$ " + str(sig), linestyle = clinestyles[i])
            i+=1
    plt.xlabel("Iterations")
    plt.ylabel("Duality gap")
    plt.title("ChambollePock accelerated, " + r"$\lambda TV(f) = $" + str(round(par,2)))
    plt.legend()
    plt.savefig("Chambolle_Acc_" + figstring, format="png", bbox_inches="tight")
    plt.clf()
    tau_list = [0.3, 0.25, 0.2, 0.1]
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    i = 0
    for tau in tau_list:
        t = time.time()
        _, list, _, iter = projected_gradient_alt_conv(f,lam, tau = tau, tol = 0.0)
        print("tau:", tau, "time", time.time() - t)
        print("duality", list[iter])
        plt.semilogy(list, label = r"$\tau =$ " + str(tau ), linestyle = linestyles[i])
        i+=1
    plt.title("Forward Backward Splitting, " + r"$\lambda TV(f) = $" + str(round(par,2)))
    plt.xlabel("Iterations")
    plt.ylabel("Duality gap")
    plt.legend()
    plt.savefig("Projected_" + figstring, format="png",bbox_inches="tight")
    plt.clf()
    pctau = 0.5
    pcsig = 0.25
    ptau = 0.2
    _, _, plist, iter = projected_gradient_alt_conv(f,lam, tau = ptau, tol = 0.0)
    #_, _, pclist1, iter = ChambollePock_denoise_conv(f,lam, tau = pctau, sig = pcsig, tol = 0.0)
    _, _, pclist2, iter = ChambollePock_denoise_conv(f,lam, tau = pctau, sig = pcsig, acc = True, tol = 0.0)
    plt.plot(plist)
    #plt.plot(pclist1)
    plt.plot(pclist2)
    plt.show()

def testOtherParameterSelections():
    u = imageio.imread('images/lenna256.jpg', pilmode = 'F')
    u = u/255.0
    f = np.copy(u)
    sigma = 0.1
    f = add_gaussian_noise(f,sigma)
    lam_init = 0.5
    q = 0.95
    u_DP, t_DP = discrepancy_ruleTV(f,sigma, lam_init = lam_init, q = q)
    u_QOC, t_QOC = quasi_optimalityTV(f, lam_init = lam_init, q = q) 
    u_LC, t_LC = L_curveTV(f, lam_init = lam_init, q = q) 

def testEstimator():
    #gridSearch(f,u)

    N = 19
    n_list = [16, 32, 64]
    for n in n_list:
        nn = n*n
        image_array = np.zeros((1,256,256))
        image_array[0] = imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
        u, f, u_hat = create_empirical_estimator(image_array, newshape = (n,n), h = 20, N_train = N, N_test = 1, sigma = 0.1, calc = False)
        #t = optTV_golden(f, u_hat, tol = 1.0e-10)
        #print("optTV lambda: ", (1-t)/t)
        plt.imsave("Estimator" + str(n) + ".png", u_hat, cmap = "gray")
        plt.imshow(u_hat, cmap = "gray")
        plt.show()
        _, R_hat = gridSearch(f, u, 100, plot = False)
        _, R_hat_list = gridSearch(f, u_hat, 100, plot = False)
        t_list = np.linspace(0,1,100)
        plt.plot(t_list[1:-1],R_hat, label = "True loss, " + r"$R(t)$")
        plt.plot(t_list[1:-1], R_hat_list, label = "Empirical loss, " + r"$\hat{R}(t)$")
        plt.title("h = 20" + ", m = " + str(n))
        plt.xlabel(r"$t$")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

def testOptTV():
    #n_list = [16, 32, 64]
    n_list = [96]
    N_list = [4, 9, 19]
    image_array = np.zeros((1,256,256))
    image_array[0] = imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    sigma = 0.1
    for n in n_list:
        for N in N_list:
            u, f, u_hat = create_empirical_estimator(image_array, newshape = (n,n), h = N, N_train = N, N_test = 1, sigma = sigma)
            _, t_opt = optTV_golden(f,u,tol = 1e-5)
            u_optTV, t = optTV_golden(f, u_hat, tol = 1e-5)
            print(n, N, np.linalg.norm(u - u_hat)/np.linalg.norm(u),np.linalg.norm(u - u_optTV)/np.linalg.norm(u), abs(t- t_opt)/t_opt) 
            plt.imsave("N" + str(N) + "n" + str(n) + ".png", u_hat, cmap = "gray")

def testParameterSelections(): 
    n = 64
    nn = n*n
    N = 20
    sigma = 0.1
    image_array = np.zeros((1,256,256))
    #image_array = np.zeros((1,512,512))
    image_array[0] = imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    u, f, u_hat = create_empirical_estimator(image_array, newshape = (n,n), h = 25, N_train = N, N_test = 1, sigma = sigma, calc = False)
    plt.imsave("lenna256.png", u, cmap = "gray")
    plt.imsave("lenna_noisy256.png", f, cmap = "gray")
    print("EMPIRICAL ESTIMATOR:")

    print("PNSR: ", psnr(u, u_hat))
    print("|u^t - u|/|u|: ", np.linalg.norm(u_hat - u)/np.linalg.norm(u))
    
    print("GRID SEARCH:")
    #t_opt,_ = gridSearch(f, u, 100, plot = False)
    u_opt, t_opt = optTV_golden(f,u, tol = 1.0e-10)
    print("|u^t - u|/|u|: ", np.linalg.norm(u_opt - u)/np.linalg.norm(u))
    print("PNSR: ", psnr(u, u_opt))
    print("OPTTV:")
    time_optTV = time.time()
    u_optTV, t_optTV = optTV_golden(f, u_hat, tol = 1.0e-6)
    print("TIME: ", time.time() - time_optTV)
    plt.imsave("optTV.png", u_optTV, cmap = "gray")
    print("PNSR: ", psnr(u, u_optTV))
    print("|u^t - u|/|u|: ", np.linalg.norm(u_optTV - u)/np.linalg.norm(u))
    print("t: ", t_optTV)
    print("|t_opt - t|/|t|: ", abs(t_opt - t_optTV)/t_opt)

    lam_init = 0.5
    q = 0.95
    
    print("DISCREPANCY PRINCIPLE:")

    time_DP = time.time()
    u_DP, t_DP = discrepancy_ruleTV(f,sigma, lam_init = lam_init, q = q)

    print("TIME: ", time.time() - time_DP)
    plt.imsave("DP.png", u_DP, cmap = "gray")
    print("|u^t - u|/|u|: ", np.linalg.norm(u_DP - u)/np.linalg.norm(u))

    print("PNSR: ", psnr(u, u_DP))
    print("t: ", t_DP)
    print("|t_opt - t|/|t|: ", abs(t_opt - t_DP)/t_opt)

    print("QUASI-OPTIMALITY CRITERION:")

    time_QOC = time.time()
    u_QOC, t_QOC = quasi_optimalityTV(f, lam_init = lam_init, q = q) 

    print("TIME: ", time.time() - time_QOC)
    plt.imsave("QOC.png", u_QOC, cmap = "gray")
    print("|u^t - u|/|u|: ", np.linalg.norm(u_QOC - u)/np.linalg.norm(u))
    print("PNSR: ", psnr(u, u_QOC))
    print("t: ", t_QOC)
    print("|t_opt - t|/|t|: ", abs(t_opt - t_QOC)/t_opt)

    print("L-CURVE:")
    time_LC = time.time()
    u_LC, t_LC = L_curveTV(f, lam_init = lam_init, q = q) 

    print("TIME: ", time.time() - time_LC)
    plt.imsave("LC.png", u_LC, cmap = "gray")
    print("|u^t - u|/|u|: ", np.linalg.norm(u_LC - u)/np.linalg.norm(u))
    print("PNSR: ", psnr(u, u_LC))
    print("t: ", t_LC)
    print("|t_opt - t|/|t|: ", abs(t_opt - t_LC)/t_opt)

def testPokemon():
    image_array = import_train()
    sigma = 0.1
    n = 64
    u, f, u_hat = create_empirical_estimator(image_array,(n,n), h = 64*64, N_train = 721, N_test = 1, sigma = sigma, calc = False)
    plt.imshow(u,cmap = "gray")
    plt.show()
    plt.imshow(f,cmap = "gray")
    plt.show()
    plt.imshow(u_hat,cmap = "gray")
    plt.show()

def createDownsampled(f,n):
    shape = f.shape
    if (not (shape[0]/n).is_integer()):
        print("CHOOSE n SMART YOU DUMB")
    m = shape[0]//n
    image_array = np.zeros((n,m,m))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                image_array[i,j,k] = f[i + j*n, i + k*n]
    return image_array

def createUpsampled(image_array, orig_shape):
    shape = image_array.shape
    image_array_upsampled = np.zeros((shape[0], ) + orig_shape)
    for i in range(shape[0]):
        #image = Image.fromarray(scipy.ndimage.gaussian_filter(image_array[i], sigma = 1.0))
        image = Image.fromarray(image_array[i])
        image = image.resize(orig_shape, Image.BICUBIC)
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

def testDownsampled():
    u =  imageio.imread('images/lenna.ppm', pilmode = 'F')/255.0
    sigma = 0.05
    f = add_gaussian_noise(u,sigma)
    orig_shape = f.shape
    n = 4
    nn = np.ceil(np.sqrt(n))
    image_array = createDownsampled(f,n)
    fig = plt.figure(figsize=(nn,nn))
    for i in range(n):
        fig.add_subplot(nn,nn,i+1)
        plt.imshow(image_array[i], cmap = "gray")
    plt.show()
    image_array_upsampled = createUpsampled(image_array,orig_shape)
    fig = plt.figure(figsize=(nn,nn))
    for i in range(n):
        fig.add_subplot(nn,nn,i+1)
        plt.imshow(image_array_upsampled[i], cmap = "gray")
    plt.show()
    #basis_img = modified_gram_schmidt_vec(image_array_upsampled)
    basis_img = modified_gram_schmidt(image_array_upsampled)
    #basis_img_img = basis_img.reshape((basis_img.shape[0],) +  orig_shape)
    fig = plt.figure(figsize=(nn,nn))
    for i in range(n):
        fig.add_subplot(nn,nn,i+1)
        #print(np.inner(basis_img[0], basis_img[i]))
        #print(np.linalg.norm(basis_img[i]))
        plt.imshow(basis_img[i], cmap = "gray")
        #plt.imshow(basis_img_img[i], cmap = "gray")
    plt.show()
    h = 4
    basis_img = basis_img[:h, :]
    N = 50
    _, t_opt = optTV_golden(f,u, tol = 1e-4)
    #t_opt = 0.9321 #256 image
    #t_opt, R_list = gridSearch(f,u,N)
    t_opt_hat, t_list,  R_list_hat = gridSearch_basis(f, basis_img, N)
    #print(t_opt, t_opt_hat)
    plt.plot(t_list, R_list_hat)
    plt.vlines(t_opt, ymin = np.min(R_list_hat), ymax = np.max(R_list_hat))
    #plt.plot(R_list)
    plt.show()

if __name__ == "__main__":
    np.random.seed(316)
    testDownsampled()
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


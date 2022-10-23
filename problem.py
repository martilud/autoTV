import autograd.numpy as np
import scipy as sp
from utils import *
class problem:
    '''
    Class for solving linear problems on the form
    (lam_l A^T A + lam_r D^T D)u = rhs
    '''
    def __init__(self, A = None, Aast = None, AA = None, D = None, Dast = None, solve_method = 'eigh'):
        self.A = A
        self.Aast = Aast
        self.AA = AA
        self.D = D
        self.Dast = Dast
        self.solve_method = solve_method
        self.set_solve_method(solve_method)

    def set_solve_method(self, solve_method):
        self.solve_method = solve_method
        if solve_method == 'solve' or solve_method == 'chol' or solve_method == 'eigh' or solve_method == 'svd':
            self.AA = np.matmul(self.A.T, self.A)
        if solve_method == 'eigh':
            self.eigvals,self.eigs = sp.linalg.eigh(self.AA)
        elif solve_method == 'svd':
            self.U, self.s, self.Vh = sp.linalg.svd(self.A) 
            self.shs = np.zeros(self.Vh.shape[0])
            self.shs[:self.s.shape[0]] = np.abs(self.s)**2
            self.ssh = np.zeros(self.U.shape[0])
            self.ssh[:self.s.shape[0]] = np.abs(self.s)**2
        elif solve_method == 'conv':
            # Calculate FFTs/DCTs...
            a = 1
    def set_pseudoinverse(self):
        if self.solve_method == "cg":
            #self.Adagger = 4.0 * self.Aast
            self.P = lambda u: 4.0 * self.Aast(self.A(u))
        else:
            self.Adagger = np.linalg.pinv(self.A)
            self.P = np.matmul(self.Adagger, self.A)
        

class parameter_problem(problem):
    def __init__(self, A, solve_method = 'eigh', parameter_method = 'dp', noise_std = 0.0, tau = 1.0):
        problem.__init__(A, solve_method)
        self.parameter_method = parameter_method
        self.noise_std = noise_std
        self.tau = tau


import numpy as np
from scipy.optimize import newton_krylov
from scipy import linalg
from scipy.fft import fft, ifft
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

# ## === --- --- === ###
#
# Solve a nonlinear ODE using implicit theta method
# solving the all-at-once system with ParaDiag
#
# dydt = f(y)
#
# ## === --- --- === ###

# parameters

alpha = 1e-0

T = 102.4
nt = 1024
theta = 0.5

dt = T/nt

q0 = 1

lamda= -0.02 + 0.3j
w = -0.25

def f(q):
    return lamda*q + w*q*q

def df(q):
    return lamda + 2*w*q

dtype = complex

time = np.linspace(dt, nt*dt, num=nt, endpoint=True)

# ## timestepping toeplitz matrices

# # mass toeplitz

# first column
b1 = np.zeros(nt, dtype=dtype)
b1[0] = 1/dt
b1[1] = -1/dt

# first row
r1 = np.zeros_like(b1)
r1[0] = b1[0]

# # function toeplitz

# first column
b2 = np.zeros(nt, dtype=dtype)
b2[0] = -theta
b2[1] = -(1-theta)

# first row
r2 = np.zeros_like(b2)
r2[0] = b2[0]

B1 = tuple((b1, r1))
B2 = tuple((b2, r2))

# ## paradiag preconditioners

# we will use scipy.optimize.newton_krylov to solve the
# nonlinear all-at-once

# update method of the preconditioner is called at
# every newton iteration with the arguments
# q (current solution vector guess)
# f (current residual vector)
# This method needs to update the time average
# with the new estimate.

class AveragedCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, b1, b2):
        self.dtype = b1.dtype
        n = len(b1)
        self.shape = tuple((n, n))

        self.b1 = b1
        self.b2 = b2

        self.l1 = fft(b1, norm='backward')
        self.l2 = fft(b2, norm='backward')

        self.qav = 0

    def _matvec(self, v):
        diag = self.l1 + self.l2*df(self.qav)
        return ifft(fft(v)/diag)

    def update(self, q, f):
        self.qav = np.sum(q)/len(q)


class AveragedAlphaCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, b1, b2, alpha=1):
        self.dtype = b1.dtype
        n = len(b1)
        self.shape = tuple((n, n))

        self.b1 = b1
        self.b2 = b2

        self.gamma = alpha**(np.arange(n)/n)

        self.l1 = fft(b1*self.gamma, norm='backward')
        self.l2 = fft(b2*self.gamma, norm='backward')

        self.qav = 0

    def _to_eigvecs(self, v):
        return fft(v*self.gamma)

    def _from_eigvecs(self, v):
        return ifft(v)/self.gamma

    def _matvec(self, v):
        diag = self.l1 + self.l2*df(self.qav)
        return self._from_eigvecs(self._to_eigvecs(v)/diag)

    def update(self, q, f):
        self.qav = np.sum(q)/len(q)


P = AveragedCirculantLinearOperator(b1, b2)

# ## right hand side

rhs = np.zeros(nt, dtype=dtype)

rhs[0] += -(b1[1] + b2[1])*q0


# ## residual of the nonlinear all-at-once system

def residual(q):
    B1q = linalg.matmul_toeplitz(B1, q)
    B2f = linalg.matmul_toeplitz(B2, f(q))
    return B1q + B2f - rhs


# ## solve all-at-once system

krylov_its = 0
newton_its = 0

# solver callbacks for nicer output


def gmres_callback(pr_norm):
    global krylov_its
    krylov_its += 1
    print(f"krylov_its: {str(krylov_its).rjust(5,' ')} | residual: {pr_norm}")
    return

def newton_callback(x, f):
    global krylov_its, newton_its

    krylov_its = 0
    newton_its += 1

    print("\n")
    print(f"newton_its: {str(newton_its).rjust(5,' ')} | residual: {linalg.norm(f)}")
    print("\n")


q = np.zeros(nt)
P.update(q, residual(q))

y = newton_krylov(residual, q, method='gmres',
                  inner_M=P,
                  callback=newton_callback,
                  inner_tol=1e-8,
                  inner_callback=gmres_callback,
                  inner_callback_type='pr_norm')

plt.plot(time, y.real)
plt.show()

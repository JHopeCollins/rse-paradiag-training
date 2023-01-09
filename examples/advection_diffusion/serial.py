
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.sparse import linalg as spla

from math import pi

### === --- --- === ###
#
# Solve the linear advection diffusion equation
#   using sequential timestepping
#
# ADE solved on a periodic mesh using:
#   time discretisation: implicit theta-method
#   space discretisation: second order central differences
#
### === --- --- === ###

# parameters

# number of time and space points
nt = 256
nx = 512

# size of domain
lx = 256.
dx = lx/nx

# velocity and reynolds number
u = 1
re = 200
cfl = 0.4

# width of initial profile
width = lx/4

# timestep
nu = width*u/re
dt = cfl*dx/u

# parameter for theta timestepping
theta=0.5

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = np.linspace( start=-lx/2, stop=lx/2, num=nx, endpoint=False )

# identity mass matrix
M = np.zeros_like(x)
M[0] = 1

# gradient matrix
Ka = np.zeros_like(x)
Ka[-1] = 1/(2*dx)
Ka[1] = -1/(2*dx)

# laplacian matrix
Kd = np.zeros_like(x)
Kd[-1] = 1/dx**2
Kd[0] = -2/dx**2
Kd[1] = 1/dx**2

Ka = Ka
Kd = Kd

K = u*Ka - nu*Kd

rhs_col = M - dt*(1 - theta)*K
lhs_col = M + dt*theta*K


# linear operators
class CirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, inverse=True):
        self.col = col
        self.shape = tuple((len(col), len(col)))

        if inverse:
            self.op = self._solve
        else:
            self.op = self._mul

        self.eigvals = fft(col, norm='backward')

    def _mul(self, v):
        return ifft(fft(v)*self.eigvals)

    def _solve(self, v):
        return ifft(fft(v)/self.eigvals)

    def _matvec(self, v):
        return self.op(v).real


rhs = CirculantLinearOperator(rhs_col, inverse=False)
lhs = CirculantLinearOperator(lhs_col, inverse=True)

# initial conditions
qinit = np.zeros_like(x)
qinit[:] = 1 + np.cos(np.minimum(2*pi*np.abs(x+lx/4)/width, pi))

q = np.zeros((nt+1, nx))
q[0] = qinit

for i in range(1, nt+1):
    q[i] = lhs.matvec(rhs.matvec(q[i-1]))

# plotting
plt.plot(x,qinit,label='i')
for i in range(20,nt+1,20):
    plt.plot(x,q[i],label=str(i))
#plt.legend(loc='center left')
plt.grid()
plt.show()

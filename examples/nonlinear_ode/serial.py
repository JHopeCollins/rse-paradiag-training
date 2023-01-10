
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

# ## === --- --- === ###
#
# Solve a nonlinear ODE using implicit theta method
# and serial timestepping
#
# dydt = f(y)
#
# ## === --- --- === ###

# parameters

nt = 1024
T = 102.4
theta = 0.5

dt = T/nt

q0 = 1

lamda= -0.02 + 0.3j
w = -0.25

def f(q):
    return lamda*q + w*q*q

def df(q):
    return lamda + 2*w*q

# setup timeseries

q = np.zeros(nt+1, dtype=complex)

q[0] = q0

# timestepping loop

def residual(qn1, qn0):
    dqdt = (qn1 - qn0)/dt
    fq = theta*f(qn1) + (1-theta)*f(qn0)
    return dqdt - fq

def jacobian(qn1, qn0):
    ddqdt = 1/dt
    dfq = theta*df(qn1)
    return ddqdt - dfq

for i in range(nt):
    q[i+1] = newton(residual, q[i], args=(q[i],),
                    fprime=jacobian)
                    #tol=1e-8, rtol=1e-8)

# plot
time = np.linspace(0, nt*dt, num=nt+1, endpoint=True)

plt.plot(time, q.real)
plt.show()

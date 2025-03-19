# ch4.py
import numpy as np
import matplotlib.pyplot as plt
from coin_flip import t

plt.close('all')

# Central Limit Theorem section (same as before)
n = 1000
trials = np.arange(1, n+1)
heads = 0
P_heads = np.ones_like(t, dtype=float)
for ii in range(n):
    if t[ii] < 0.5:
        heads += 1
    P_heads[ii] = heads / (ii + 1)
plt.figure()
for ii in range(n):
    plt.clf()
    plt.plot(trials[:ii+1], P_heads[:ii+1])
    plt.ylim([0, 1])
    plt.xlabel('Trial Number')
    plt.ylabel('Probability of Heads after n trials')
    plt.title('Trial Number vs Percent Heads')
    plt.pause(0.001)

# Ex 4.1: Animated plots for sequences

n_vals = np.arange(1, 51)

# 1. A_n = 2 - 1/(n^2)
A_n = 2 - 1/(n_vals**2)
plt.figure()
for ii in range(len(n_vals)):
    plt.clf()
    plt.plot(n_vals[:ii+1], A_n[:ii+1])
    plt.ylim([1, 2])
    plt.xlabel('Trial Number')
    plt.ylabel('A_n')
    plt.title('Trial Number vs A_n')
    plt.pause(0.1)

# 2. B_n = (n^2 + 1)/n
B_n = (n_vals**2 + 1) / n_vals
plt.figure()
for ii in range(len(n_vals)):
    plt.clf()
    plt.plot(n_vals[:ii+1], B_n[:ii+1])
    plt.ylim([0, 60])
    plt.xlabel('Trial Number')
    plt.ylabel('B_n')
    plt.title('Trial Number vs B_n')
    plt.pause(0.1)

# 3. C_n = (-1)^n * (1 - (1/n))
C_n = ((-1)**n_vals) * (1 - 1/n_vals)
plt.figure()
for ii in range(len(n_vals)):
    plt.clf()
    plt.plot(n_vals[:ii+1], C_n[:ii+1])
    plt.ylim([-1, 1])
    plt.xlabel('Trial Number')
    plt.ylabel('C_n')
    plt.title('Trial Number vs C_n')
    plt.pause(0.1)

# Fig 4.3: Animated plots for functions

x = np.arange(-50, 51, 1)
# f(x) = sqrt(x) (only for x>=0; for x<0, take 0)
f_x = np.sqrt(np.maximum(x, 0))
plt.figure()
for ii in range(len(x)):
    plt.clf()
    plt.plot(x[:ii+1], f_x[:ii+1])
    plt.ylim([0, 10])
    plt.xlabel('Trial Number')
    plt.ylabel('f(x)')
    plt.title('Trial Number vs f(x)')
    plt.pause(0.05)

# f(x) = 1/x (avoid division by zero)
f_x = 1 / x.astype(float)
f_x[np.isinf(f_x)] = np.nan
plt.figure()
for ii in range(len(x)):
    plt.clf()
    plt.plot(x[:ii+1], f_x[:ii+1])
    plt.ylim([-1, 1])
    plt.xlabel('Trial Number')
    plt.ylabel('f(x)')
    plt.title('Trial Number vs f(x)')
    plt.pause(0.05)

# Ex 4.4: f(x) = sqrt(x)
x = np.arange(1, 51)
f_x = np.sqrt(x)
plt.figure()
for ii in range(len(x)):
    plt.clf()
    plt.plot(x[:ii+1], f_x[:ii+1])
    plt.ylim([0, 10])
    plt.xlabel('Trial Number')
    plt.ylabel('f(x)')
    plt.title('Trial Number vs f(x) = sqrt(x)')
    plt.pause(0.05)

# f(x) = e^x
x = np.arange(-5, 6, 1)
f_x = np.exp(x)
plt.figure()
for ii in range(len(x)):
    plt.clf()
    plt.plot(x[:ii+1], f_x[:ii+1])
    plt.ylim([0, 150])
    plt.xlabel('Trial Number')
    plt.ylabel('f(x)')
    plt.title('Trial Number vs f(x) = e^x')
    plt.pause(0.05)

# f(x) = 1 + (1/x^2) (be cautious at x=0)
x = np.arange(-2, 2.05, 0.05)
f_x = 1 + 1/(x**2)
plt.figure()
for ii in range(len(x)):
    plt.clf()
    plt.plot(x[:ii+1], f_x[:ii+1])
    plt.ylim([0, 150])
    plt.xlabel('Trial Number')
    plt.ylabel('f(x)')
    plt.title('Trial Number vs f(x) = 1+(1/x^2)')
    plt.pause(0.05)

# f(x) = floor(x)
x = np.arange(0, 6, 0.1)
f_x = np.floor(x)
plt.figure()
for ii in range(len(x)):
    plt.clf()
    plt.plot(x[:ii+1], f_x[:ii+1], '*')
    plt.ylim([0, 5])
    plt.xlabel('Trial Number')
    plt.ylabel('f(x)')
    plt.title('Trial Number vs f(x) = floor(x)')
    plt.pause(0.05)

plt.show()

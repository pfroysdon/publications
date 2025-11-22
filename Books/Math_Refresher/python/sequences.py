# sequences.py
# Sequences
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

n = 20
x = np.arange(1, n+1)

An = np.zeros((n,))
Bn = np.zeros((n,))
Cn = np.zeros((n,))

for ii in range(1, n+1):
    An[ii-1] = 2 - (1 / (ii ** 2))
    Bn[ii-1] = (ii**2 + 1) / ii
    Cn[ii-1] = ((-1) ** ii) * (1 - (1 / ii))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, An, '.', linewidth=1)
plt.plot(x, An, 'b')
plt.xlabel(r'$$n$$', fontsize=12)
plt.ylabel(r'$$A_n$$', fontsize=12)
plt.title(r'$$A_n$$', fontsize=12)

plt.subplot(1, 3, 2)
plt.plot(x, Bn, '.', linewidth=1)
plt.plot(x, Bn, 'b')
plt.xlabel(r'$$n$$', fontsize=12)
plt.ylabel(r'$$B_n$$', fontsize=12)
plt.title(r'$$B_n$$', fontsize=12)

plt.subplot(1, 3, 3)
plt.plot(x, Cn, '.', linewidth=1)
plt.plot(x, Cn, 'b')
plt.xlabel(r'$$n$$', fontsize=12)
plt.ylabel(r'$$C_n$$', fontsize=12)
plt.title(r'$$C_n$$', fontsize=12)

plt.savefig("../figures/sequences.pdf")
plt.show()

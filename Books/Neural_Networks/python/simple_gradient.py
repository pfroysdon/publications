import numpy as np
import matplotlib.pyplot as plt
      
x_old = 0 # The value does not matter as long as abs(x_new - x_old) > precision
x_new = 6 # The algorithm starts at x=6
gamma = 0.01 # step size
precision = 0.00001

def f(x):
    y = x**4 - 3*x**2 + 2
    return y
    
def df(x):
	y = 4 * x**3 - 9 * x**2
	return y

while abs(x_new - x_old) > precision:
	x_old = x_new
	x_new += -gamma * df(x_old)

print("The local minimum occurs at %f" % x_new)

n = 10
x = np.linspace(1, 10, n)
y = np.zeros(n)
for i in range(n):
    y[i] = f(x[i])
 
plt.figure(1)    
plt.plot(x, y, 'o-',label='f(x)')
plt.plot(x_new, f(x_new), 'r*',label='optimal f(x)')
plt.legend()
plt.title('f(x) = x^4 - 3x^2 + 2')
plt.grid(True)
plt.show()
plt.savefig('gradient_descent.png')
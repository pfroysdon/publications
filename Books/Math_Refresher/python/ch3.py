# ch3.py
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix  # To compute reduced row echelon form if needed

plt.close('all')

# Vectors
v = np.array([1, 2, 3])
print("Row vector v:", v)

v = np.array([[1], [2], [3]])
print("Column vector v:\n", v)

u = np.array([1, 1, 1])
v = np.array([2, 2, 2])
result_1 = u + v
print("u + v =", result_1)

result_1 = u - v
print("u - v =", result_1)

c = 2
result_1 = c * v
print("c * v =", result_1)

result_1 = np.dot(u, v)
print("dot(u, v) =", result_1)

result_2 = np.dot(u, v)  # equivalent to u*v' in MATLAB for these vectors
print("u * v' =", result_2)

# Matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[1, 2, 1],
              [2, 1, 2]])
result_1 = A + B
print("A + B =\n", result_1)

s = 2
result_1 = s * A
print("s*A =\n", result_1)

result_2 = A.dot(B.T)
print("A * B' =\n", result_2)

# Laws of Matrix Algebra
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
B = 2 * np.ones((3, 3))
C = 5 * np.ones((3, 3))
# Associative property
result_1 = (A + B) + C
result_2 = A + (B + C)
print("Associative addition:", np.allclose(result_1, result_2))

result_1 = (A.dot(B)).dot(C)
result_2 = A.dot(B.dot(C))
print("Associative multiplication:", np.allclose(result_1, result_2))

# Commutative property (addition is commutative)
result_1 = A + B
result_2 = B + A
print("Commutative addition:", np.allclose(result_1, result_2))

# Distributive property
result_1 = A.dot(B + C)
result_2 = A.dot(B) + A.dot(C)
print("Distributive property 1:", np.allclose(result_1, result_2))

result_1 = (A + B).dot(C)
result_2 = A.dot(C) + B.dot(C)
print("Distributive property 2:", np.allclose(result_1, result_2))

# Order matters (matrix multiplication is not commutative)
result_1 = A.dot(B)
result_2 = B.dot(A)
print("A*B equals B*A?", np.allclose(result_1, result_2))

# Transpose examples
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("A':\n", A.T)
a = np.array([1, 2, 3])
print("a (as a 1D array, transpose does not change shape):", a)

# Transpose properties
result_1 = (A + B).T
result_2 = A.T + B.T
print("Transpose property (A+B)' =", np.allclose(result_1, result_2))
result_1 = A.T.T
result_2 = A
print("Double transpose equals original:", np.allclose(result_1, result_2))
result_1 = (s*A).T
result_2 = s*A.T
print("Transpose of s*A equals s*A':", np.allclose(result_1, result_2))
result_1 = (A.dot(B)).T
result_2 = B.T.dot(A.T)
print("Transpose of product equals product of transposes reversed:", np.allclose(result_1, result_2))

# System of Linear Equations
# Solve:
#   x - 3y = -3
#   2x + y = 8
A = np.array([[1, -3],
              [2,  1]])
b = np.array([-3, 8])
solution = np.linalg.solve(A, b)
print("Solution using np.linalg.solve:", solution)

x_sol = np.linalg.inv(A).dot(b)
print("Solution using inverse:", x_sol)

# Properties of the Inverse
A = np.array([[1, 1, 1],
              [0, 2, 3],
              [5, 5, 1]])
B = np.array([[1, 0, 4],
              [0, 2, 0],
              [0, 0, 1]])
result_1 = A.dot(np.linalg.inv(A))
result_2 = np.linalg.inv(A).dot(A)
print("A * inv(A) =\n", result_1)
print("inv(A) * A =\n", result_2)
result_1 = np.linalg.inv(np.linalg.inv(A))
print("inv(inv(A)) =\n", result_1)
result_1 = np.linalg.inv(A.dot(B))
result_2 = np.linalg.inv(B).dot(np.linalg.inv(A))
print("inv(A*B) =\n", result_1)
print("inv(B)*inv(A) =\n", result_2)

# Linear Systems using inverses
b = np.array([10, 50, 10])
x_sol = np.linalg.inv(A).dot(b)
print("Solution for Ax=b:", x_sol)

# Determinants
det_A = np.linalg.det(A)
print("Determinant of A:", det_A)

import numpy as np
from demoFcns import f, simple_looped_nn_calc, matrix_feed_forward_calc

# weights between layer 1 and 2
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
w2 = np.zeros((1, 3))
w2[0,:] = np.array([0.5, 0.5, 0.5])	

# bias between layer 1 and 2
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

# vectorize		
w = [w1, w2]
b = [b1, b2]

#a dummy x input vector
x = [1.5, 2.0, 3.0]

# call the main function & print result
result1 = simple_looped_nn_calc(3, x, w, b)
print('simple_looped_nn_calc =', result1)
    
result2 = matrix_feed_forward_calc(3, x, w, b)
print('matrix_feed_forward_calc =', result2)
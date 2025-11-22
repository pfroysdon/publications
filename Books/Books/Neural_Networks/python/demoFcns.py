import numpy as np

def f(x):
	return 1 / (1 + np.exp(-x))

def simple_looped_nn_calc(n_layers, x, w, b):
	for l in range(n_layers-1):
		#Setup the input array which the weights will be multiplied by for each layer
		#If it's the first layer, the input array will be the x input vector
		#If it's not the first layer, the input to the next layer will be the 
		#output of the previous layer
		if l == 0:
			node_in = x
		else:
			node_in = h
		#Setup the output array for the nodes in layer l + 1
		h = np.zeros((w[l].shape[0],))
		#loop through the rows of the weight array
		for i in range(w[l].shape[0]):
			#setup the sum inside the activation function
			f_sum = 0
			#loop through the columns of the weight array
			for j in range(w[l].shape[1]):
				f_sum += w[l][i][j] * node_in[j]
			#add the bias
			f_sum += b[l][i]
			#finally use the activation function to calculate the
			#i-th output i.e. h1, h2, h3
			h[i] = f(f_sum)
	return h

def matrix_feed_forward_calc(n_layers, x, w, b):
	for l in range(n_layers-1):
		if l == 0:
			node_in = x
		else:
			node_in = h
		z = w[l].dot(node_in) + b[l]
		h = f(z)
	return h
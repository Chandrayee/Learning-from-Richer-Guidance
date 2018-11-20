import numpy as np
import math
import itertools as it

#ground truth weights
#w_i = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
w_i = np.array([-1,-0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
#w_i = np.array([0.25, 0.5, 0.75, 1.])
r = 7
W = [i for i in it.product(w_i, repeat = r)]
W = np.array(W)
W = W[np.where((W[:,0] >= 0.25) & (np.absolute(W[:,1]) < 1.) & (W[:,2] >= 0.25) & (W[:,3] <= -0.75) & (np.sign(W[:,6]) == -1) & (np.absolute(W[:,6]) <= 0.5))]
weights =     [[0.25,  0.25, 0.25, -1.,    0.25, -0.75, -0.25], 
			   [0.25, -0.25, 0.25, -1.,   -1.,    0.25, -0.25],
			   [0.25, -0.25, 0.75, -0.25, -1.,    0.25, -0.25],
			   [0.25, -0.75, 0.25, -1.,   -1.,   -1.,   -0.25], 
			   [0.25, -0.75, 0.25, -1.,   -1.,    1.,   -0.25], 
			   [0.75, -0.75, 0.25, -1.,   -1.,    0.25, -0.25], 
			   [0.25,  0.25, 0.25,  1.,    0.25,  0.25, -0.5 ],
			   [0.25,  0.25, 0.75, -1.,    0.25,  0.75,  0.25],  
			   [0.25,  0.25, 0.75, -1.,   -0.75,  0.25, -0.25], 
			   [0.25,  0.25, 0.75, -1.,    0.75,  0.25, -0.5 ] ]
			   

W = list(W)
W.append(weights[2])
W.append(weights[6])
W.append(weights[7])
	
W = np.array(W)

indices = np.random.randint(len(W), size=12000)
indices = list(indices)

for i in range(len(weights)):
	index = np.where(np.all(W == weights[i], axis = 1))
	print index[0][0], " : ", W[index[0][0]]
	if index not in indices:
		indices.append(index[0][0])
		
indices = np.array(indices)


#np.save('indices12000.npy',indices)
	
#print W.shape[0]

# X = []
# for i in range(50):
# 		x = np.random.normal(size = r)
# 		X.append(x/np.linalg.norm(x))
#print('\nnew data is generated.')








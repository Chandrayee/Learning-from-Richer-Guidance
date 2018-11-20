import pickle
from world import *
import numpy as np
import math 

train_phis = np.load('trialdata2.npy')
fn = 'newdata.pkl'
with open(fn, 'r') as out:
	X = pickle.load(out)
count = 0
for i in range(len(X)):	
	humanstates = X[i][-1].human['A'].x
	pos = [x[1] for x in humanstates]
	sum1 = 0
	for k in range(len(pos)-1):
		diff = pos[k+1] - pos[k]
		print 'diff', diff
		print math.exp(-(min(diff, 0.0))**2)
		sum1 = sum1 + math.exp(-(min(diff, 0.0))**2)
	print 'next'	
	humanstates = X[i][-1].human['B'].x
	pos = [x[1] for x in humanstates]
	sum2 = 0
	for k in range(len(pos)-1):
		diff = pos[k+1] - pos[k]
		print 'diff', diff
		print math.exp(-(min(diff, 0.0))**2)
		sum2 = sum2 + math.exp(-(min(diff, 0.0))**2)
	if sum1 == sum2:
		if train_phis[i][-1] != 0.0:
			print 'bummer'
			count += 1
		train_phis[i][-1] = 0.0
		

print count

np.save('trialdata2_revcorrected.npy', train_phis)
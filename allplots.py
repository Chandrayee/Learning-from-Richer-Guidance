import pickle
import numpy as np
import matplotlib.pyplot as plt

def initialize(fn):
	with open(fn, 'rb') as out:
		data = pickle.load(out)
	return data
	
def truereward(W, optphis, index):
	return np.dot(W, optphis[index])

def plotprob(index, data, colour, label):	
	prob = []
	for i in range(len(data)):
		p = data[str(i)]['prob'][:40]
		prob.append([p[k][index] for k in range(len(p))])
		
	prob = np.transpose(np.array(prob))
	meanprob = np.mean(prob, axis = 1)
	errprob = np.std(prob, axis = 1)/np.sqrt(len(data))
	plt.plot(np.arange(40), meanprob, colour, linewidth = 3., label = label)
	plt.errorbar(np.arange(40), meanprob, yerr=errprob, fmt = 'o', linewidth = 1., color = colour, markeredgewidth = 0.0)
	
	
def plotreward(index, data, colour, label):	
	runreward = []
	n = len(data)
	for i in range(len(data)):
		indices = [np.argmax(k) for k in data[str(i)]['prob']]
		envreward = []
		for j in range(10):
			filename = 'testndf' + str(j+1) + '.npy'
			test_phis = np.load(filename)
			test_phis = [list(x[0]) for x in test_phis]
			test_phis = np.array(test_phis)
			test_phis = test_phis[newindex]
			reward = []
			for ind in indices:
				reward.append(truereward(W, test_phis, ind))
			envreward.append(np.array(reward))
		envreward = np.mean(np.transpose(np.array(envreward)), axis = 1)
		runreward.append(envreward)
	runreward = np.transpose(np.array(runreward))
	meanreward = np.mean(runreward, axis = 1)
	errreward = np.std(runreward, axis = 1)/np.sqrt(n)
	plt.plot(np.arange(40), meanreward, colour, linewidth = 3., label = label)
	plt.errorbar(np.arange(40), meanreward, yerr=errreward, fmt = 'o', linewidth = 1., color = colour, markeredgewidth = 0.0)
	
def plottruereward(index):
	trueR = []
	for i in range(10):
		filename = 'testndf' + str(i+1) + '.npy'
		test_phis = np.load(filename)
		test_phis = [list(x[0]) for x in test_phis]
		test_phis = np.array(test_phis)
		test_phis = test_phis[newindex]
		trueR.append(truereward(W, test_phis, index))
	trueR = np.mean(trueR)
	plt.plot(np.arange(40), trueR * np.ones(50), '--k', linewidth = 2., label = 'true reward')

ind = [12001, 8556, 12002,  827, 5416, 12003, 11308, 4668, 337, 8416, 1566]
ind = [0, 4, 5, 6, 7, 8, 9, 11, 12, 13, 1500, 1773, 816, 1589, 541, 51, 600, 49, 1759]
index = 1500	
#newindex = np.load('indices12000.npy')
M = np.load('weights2000.npy')
M = np.array([k/np.linalg.norm(k) for k in M])
#M = M[newindex]
W = M[index]
	

fn = 'baselinenoisy_' + str(index) + '.pkl'
data = initialize(fn)
plotprob(index, data, '#696969', 'baseline')

fn = 'cirlfnoisy_' + str(index) + '.pkl'
data = initialize(fn)
plotprob(index, data, 'g', 'CIRLF')

plt.legend(loc = 4)
plt.xlabel('iterations')
plt.ylabel(r'p($\theta_{GT}$)')
plt.savefig('prob_noisy_1500.png')


# plottruereward(index)
# fn = 'baselinenoisy_' + str(index) + '.pkl'
# data = initialize(fn)
# plotreward(index, data, 'r', 'baseline')
#  
# fn = 'cirlfnoisy_' + str(index) + '.pkl'
# data = initialize(fn)
# plotreward(index, data, 'g', 'CIRLF')
# plt.legend(loc = 4)
# plt.xlabel('iterations')
# plt.ylabel(r'Value of $\phi^*(\hat{\theta})$ w.r.t $\theta_{GT}$')
# plt.savefig('reward_noisy_5416.png')

print "done"






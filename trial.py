from world import world
import numpy as np
import shelve
import sys
import getopt
from utils import vector, matrix, Maximizer
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
from visual import select
import matplotlib.pyplot as plt
import time
from scipy import stats

def oracle(w, W, df, cirlf, bm1, bm2, ind_feature):
	if not cirlf: 
		print "baseline"
		if np.dot(W,df) > 0: 
			p_phis = 1/(1+np.exp(bm1 * -np.dot(w,df))) 
		elif np.dot(W,df) < 0:
			p_phis = 1/(1+np.exp(bm1 * np.dot(w,df)))
	else:
		print "cirlf"
		if np.dot(W,df) > 0: 
			p_phis = 1/(1+np.exp(bm1 * -np.dot(w,df))) * np.exp(bm2 * np.absolute(w[:,ind_feature]*df[ind_feature]))/np.sum(np.exp(bm2 * np.absolute(w * df)), axis = 1)
		elif np.dot(W,df) < 0:
			p_phis = 1/(1+np.exp(bm1 * np.dot(w,df))) * np.exp(bm2 * np.absolute(w[:,ind_feature]*df[ind_feature]))/np.sum(np.exp(bm2 * np.absolute(w * df)), axis = 1)
	return p_phis
	
if __name__=='__main__':
	optlist, args = getopt.gnu_getopt(sys.argv, 'n:m:r:')
	opts = dict(optlist)
	N = int(opts.get('-n', 50))
	method = int(opts.get('-m', 1))
	run = int(opts.get('-r', 0))
	w = np.array([[ 0.25, -0.75,  0.25, -1.  , -1.  , -1.  , -0.25],[ 0.25,  0.25,  0.25, -1.  ,  0.25, -0.75, -0.25],[ 0.25, -0.25,  0.25, -1.  , -1.  ,  0.25, -0.25],[ 0.25,  0.25,  0.75, -1.  ,  0.25,  0.75,  0.25]])
	print w
	W = w[1]
	train_phis = []
	for i in range(30000):
		world.randomize()
		train_phis.append(world.mdf)
	train_phis = np.array(train_phis)
	for i in range(w.shape[1]):
		print "stats", [np.mean(train_phis[:,i]), np.std(train_phis[:,i])]	
	
	if method == 1:
		print "\nRunning method 13: probabilistic cirlf"
		beta1_model = 2.061
		beta2_model = 1.465
		beta1_sim = 2.061
		beta2_sim = 1.465
		cand_phis = train_phis
		p_phis = np.ones(w.shape[0])
		p_w = np.ones(w.shape[0])
		locations = np.arange(len(train_phis))
		indices = []
		prob = []
		oracledata = []
		ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
		pfeat = np.exp(beta2_sim * np.absolute(W*cand_phis))
		sumarray = np.sum(pfeat, axis = 1)
		for k in range(len(sumarray)):
			pfeat[k] = pfeat[k]/sumarray[k]	
		wdotx = np.dot(w,np.transpose(cand_phis))
		z1 = 1/(1+np.exp(-beta1_model * wdotx))
		z2 = 1/(1+np.exp(beta1_model * wdotx))
		
		for j in range(N):
			p_w = p_w * p_phis
			p_w = p_w/np.sum(p_w)
			print "\niteration no: ", j
			print "weight update: ", p_w
			obj = np.zeros(len(cand_phis))
			for i in range(len(cand_phis)):
				x = cand_phis[i]
				den = np.sum(np.exp(beta2_model * np.absolute(w*x)), axis = 1)
				remA = 0.0
				remB = 0.0
				for n in range(w.shape[1]):
					num = np.exp(beta2_model * np.absolute(w[:,n]*x[n]))
					feat = num/den
					choiceA = z1[:, i] * feat
					choiceB = z2[:, i] * feat
					remA = remA + choiceA*(1. - choiceA)
					remB = remB + choiceB*(1. - choiceB)
				obj[i] = np.dot(p_w, remA + remB)
			df = cand_phis[np.argmax(obj)]
			ind_opt = np.argmax(obj)
			
				
			#simulate oracle choice
			truefeature = np.argmax(np.absolute(W * df))
			if np.dot(W, df) > 0:
				oracledata.append(['A', truefeature])
			else:
				oracledata.append(['B', truefeature])
				
			#ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
			p_phis = oracle(w, W, df, True, beta1_model, beta2_model, truefeature)
			
			#simulate noisy human choice
			#pick = np.random.binomial(1, ppref[locations[ind_opt]])
			#ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
			#p_phis = noisy(w, df, True, pick, beta1_model, beta2_model, ind_feature)
			
					
			cand_phis = np.delete(cand_phis, ind_opt, 0)
			z1 = np.delete(z1, ind_opt, 1)
			z2 = np.delete(z2, ind_opt, 1)
			indices.append(locations[ind_opt])
			locations = np.delete(locations, ind_opt)
			prob.append(p_w)
		
		data = np.array([k[index] for k in prob])
		plt.plot(data, 'g')
	
	
	method = 2
	if method == 2:
		print "\nRunning method 14: probabilistic baseline"
		beta1_model = 2.061
		beta1_sim = 2.061
		cand_phis = train_phis
		locations = np.arange(len(train_phis))
		p_phis = np.ones(w.shape[0])
		p_w = np.ones(w.shape[0])
		indices = []
		prob = []
		oracledata = []
		ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
		wdotx = np.dot(w,np.transpose(cand_phis))
		z1 = 1/(1+np.exp(-beta1_model * wdotx))
		z2 = 1/(1+np.exp(beta1_model * wdotx))
		for j in range(N):
			p_w = p_w * p_phis
			p_w = p_w/np.sum(p_w)
			print "\niteration no: ", j
			print "weight update: ", p_w
			obj = np.dot(np.transpose(p_w), (z1 * z2)**2)
			df = cand_phis[np.argmax(obj)]
			ind_opt = np.argmax(obj)
			
			#simulate oracle choice
			p_phis = oracle(w, W, df, False, beta1_model, 0, 0)
			
			if np.dot(W, df) > 0:
				oracledata.append(['A', np.argmax(np.absolute(W * df))])
			else:
				oracledata.append(['B', np.argmax(np.absolute(W * df))])
			
			#noisy human choice
			#pick = np.random.binomial(1, ppref[locations[ind_opt]])
			#p_phis = noisy(w, df, False, pick, beta1_model, 0., 0.)
	 			
			cand_phis = np.delete(cand_phis, ind_opt, 0)
			z1 = np.delete(z1, ind_opt, 1)
			z2 = np.delete(z2, ind_opt, 1)
			indices.append(locations[ind_opt])
			locations = np.delete(locations, ind_opt)
			prob.append(p_w)
						
			#stopping criterion
			#if np.max(p_w) > 0.1:
				#break	
		
		data = np.array([k[index] for k in prob])
		plt.plot(data, 'r')	
		plt.show()	
#!/usr/bin/env python

#################################################################################

# project: Preference-learning with Rich Queries
# date: 10/22/17
# author: Chandrayee Basu
# description: main algorithm
# This file implements three methods: 
# method 1 is probabilistic preference learning with feature query
# method 2 is probabilistic preference learning without feature query
# method 3 is probabilistic preference learning with feature query with possible
# response skip
# To run this program with default settings call "python testDrive"
# To run this program with arguments call "python testDrive -n 50 -m 1 -r 0 -h 0"
# Arguments: n = # of iterations, m = method, r = run # for the same 
# condition in case of noisy human input, h = whether human is noisy or oracle

#################################################################################

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
import random
import pickle
import math

def pickrandom(x):
	randind = np.random.randint(len(x), size = 1000)
	return randind

def rewardloss(W, trueW, trueR, testphis, p_w = []):	
	if p_w == []:
		return np.linalg.norm(trueR - np.average([np.dot(trueW, k) for k in testphis]))
	else:
		if len(W) == 7:
			return np.linalg.norm(trueR - np.dot(trueW, testphis))
		else:
			return np.linalg.norm(trueR - np.sum(p_w * [np.dot(trueW, k) for k in testphis]))
		
def truereward(W, optphis, index):
	return np.dot(W, optphis[index])
		
	
def randomloss(x, z, p, k):
	indices = pickrandom(x)
	return rewardloss(x[indices],z, p, k[indices])	
	
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
	
def noisy(w, df, cirlf, pick, bm1, bm2, ind_feature):
	if not cirlf:
		print "baseline"
		if pick == 1:
			p_phis = 1/(1+np.exp(bm1 * -np.dot(w,df))) 
		elif pick == 0:
			p_phis = 1/(1+np.exp(bm1 * np.dot(w,df)))	
	else:
		print "cirlf"
		if pick == 1:
			p_phis = 1/(1+np.exp(bm1 * -np.dot(w,df))) * np.exp(bm2 * np.absolute(w[:,ind_feature]*df[ind_feature]))/np.sum(np.exp(bm2 * np.absolute(w * df)), axis = 1)
		elif pick == 0: 
			p_phis = 1/(1+np.exp(bm1 * np.dot(w,df))) * np.exp(bm2 * np.absolute(w[:,ind_feature]*df[ind_feature]))/np.sum(np.exp(bm2 * np.absolute(w * df)), axis = 1)
	return p_phis
	
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
	plt.plot(np.arange(50), trueR * np.ones(50), '--k', linewidth = 2., label = 'true reward')
	
def plotoracleprob(index, file, colour, label):
	file = np.load(file)
	prob = np.array([k[index] for k in file])
	print np.argmax(prob)
	plt.plot(prob, colour, linewidth = 3., label = label)
	plt.xlabel('iterations')
	plt.ylabel('Probability of true weight')
	plt.legend(loc = 2)
	
def plotnormdiff(index, file, colour, label):
	file = np.load(file)
	prob = np.array([k[index] for k in file])
	indices = [np.argmax(k) for k in np.array(file)]
	normdiff = [np.linalg.norm(W - k) for k in M[indices]]
	for i in range(len(indices)):
		print [M[indices[i]], W, normdiff[i]]
	print 'done'
	plt.plot(normdiff, colour, linewidth = 3., label = label)
	plt.xlabel('iterations')
	plt.ylabel('Normed diff')
	plt.legend(loc = 2)
	
def plotnoisyprob(index, file, colour, label, n):
	prob = []
	for run in np.arange(n):
		file = np.load(file + str(run) + '.npy')
		prob.append(np.array([k[index] for k in file]))
	prob = np.transpose(np.array(prob))
	meanprob = np.mean(prob, axis = 1)
	errprob = np.std(prob, axis = 1)/np.sqrt(n)
	plt.plot(np.arange(50), meanprob, colour, linewidth = 3., label = label)
	plt.errorbar(np.arange(50), meanprob, yerr=errprob, fmt = 'o', linewidth = 1., color = colour, markeredgewidth = 0.0)	
	
def plotoraclereward(index, file, colour, label):
	file = np.load(file)
	genreward = []
	for i in range(10):
		filename = 'testndf' + str(i+1) + '.npy'
		test_phis = np.load(filename)
		test_phis = [list(x[0]) for x in test_phis]
		test_phis = np.array(test_phis)
		test_phis = test_phis[newindex]
		indices = [np.argmax(k) for k in np.array(file)]
		reward = []
		for ind in indices:
			reward.append(truereward(W, test_phis, ind))
		genreward.append(np.array(reward))
		
	genreward = np.transpose(np.array(genreward))
	genmean = np.mean(genreward, axis = 1)
	generror = np.std(genreward, axis = 1)/np.sqrt(10.)
	plt.plot(np.arange(50), genmean, colour, linewidth = 3., label = label)
	plt.errorbar(np.arange(50), genmean, yerr=generror, fmt = 'o', linewidth = 1., color = colour, markeredgewidth = 0.0)
	plt.legend(loc = 4)
	plt.xlabel('iterations')
	plt.ylabel('Algorithm reward')
	
	
def plotnoisyreward(index, file, colour, label, n):
	runreward = [] 
	for run in np.arange(n):
		file = np.load(file + str(run) + '.npy')
		indices = [np.argmax(k) for k in np.array(file)]
		envreward = []
		for i in range(10):
			filename = 'testndf' + str(i+1) + '.npy'
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
	plt.plot(np.arange(50), meanreward, colour, linewidth = 3., label = label)
	plt.errorbar(np.arange(50), meanreward, yerr=errreward, fmt = 'o', linewidth = 1., color = colour, markeredgewidth = 0.0)
	plt.legend(loc = 4)
	plt.xlabel('iterations')
	plt.ylabel('Algorithm reward')
	
def increasedata(x, fn):
	x = list(x)
	moredata = np.load(fn)
	for i in range(len(moredata)):
		x.append(moredata[i])
	x = np.array(x)
	return x
	
def generaterandomdata(num):
	x = []
	for i in range(num):
		world.randomize()
		x.append(world.mdf)
	x = np.array(x)
	return x
		 		
	
if __name__=='__main__':
	optlist, args = getopt.gnu_getopt(sys.argv, 'n:m:r:h:')
	opts = dict(optlist)
	N = int(opts.get('-n', 40))
	method = int(opts.get('-m', 1))
	run = int(opts.get('-r', 0))
	noise = int(opts.get('-h', 0))
	
	#load query pool
	train_phis = np.load('corrmdf.npy')
	train_phis = np.array([k/max(1, np.linalg.norm(k)) for k in train_phis])

	#load discrete weight space
	M = np.load('weights2000.npy')
	M = list(M)
	M = np.array([k/np.linalg.norm(k) for k in M])
	
	#load trajectory pairs corresponding to the query pool
	fn = 'corrdata.pkl'
	with open(fn, 'r') as out:
		X = pickle.load(out)

	#array of indices in M indicating location of sample ground truth weights (optional)
	ind = [1759, 600, 12, 541, 7, 49, 9, 13, 4, 8, 704, 206, 165, 1971, 81, 902, 1950, 180, 822, 977]
	
	#keep track of start time
	start = time.time()
	
#####################################   METHOD 1 #################################################
		
	if method == 1:
		print "\nRunning method 1: probabilistic comparison learning with feature query"
		
		for index in ind[0]:
			
			####################### INPUTS ############################################
			
			#temporary variables
			indices = []
			optindices = []
			prob = []
			oracledata = []
			
			#model parameters: rationality coefficients
			beta1_model = 100.
			beta2_model = 2.5
			beta1_sim = 100.
			beta2_sim = 0.0001
			
			#pool, ground truth reward, weight space
			W = M[index]
			cand_phis = train_phis
			locations = np.arange(len(train_phis))
			w = M.copy()
			
			###################### ALGORITHM #########################################
			
			p_phis = np.ones(w.shape[0]) # update function
			p_w = np.ones(w.shape[0]) # initial probability distribution over weights 
			ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))] #preference model
			pfeat = np.exp(beta2_sim * np.absolute(W*cand_phis)) #feature selection model
			sumarray = np.sum(pfeat, axis = 1)
			for k in range(len(sumarray)):
				pfeat[k] = pfeat[k]/sumarray[k]	
			wdotx = np.dot(w,np.transpose(cand_phis))
			z1 = 1/(1+np.exp(-beta1_model * wdotx))
			z2 = 1/(1+np.exp(beta1_model * wdotx))

			for j in range(10):
				if len(optindices) >= 1:
					data = np.delete(data, optindices[-1], 0)
					z1 = np.delete(z1, optindices[-1], 1)
					z2 = np.delete(z2, optindices[-1], 1)
					locations = np.delete(locations, optindices[-1])
				else:
					data = cand_phis
							
				p_w = p_w * p_phis
				p_w = p_w/np.sum(p_w)
				obj = np.zeros(len(data))
				for i in range(len(data)):
					x = data[i]
					den = np.sum(np.exp(beta2_model * np.absolute(w*x)), axis = 1)
					remA = 0.0
					remB = 0.0
					for n in range(w.shape[1]):
						num = np.exp(beta2_model * np.absolute(w[:,n]*x[n]))
						feat = num/den
						choiceA = z1[:, i] * feat
						choiceB = z2[:, i] * feat
						remA = remA + np.dot(p_w, choiceA)*np.dot(p_w, 1 - choiceA)
						remB = remB + np.dot(p_w, choiceB)*np.dot(p_w, 1 - choiceB)
					obj[i] = remA + remB
				df = data[np.argmax(obj)]
				ind_opt = np.argmax(obj)
			
				#visual check for the algorithm 		
				'''humanstates = X[originalind][-1].human['A'].x
				pos = [x[1] for x in humanstates]
				for k in range(len(pos)-1):
					diff = pos[k+1] - pos[k]
					print 'diff', diff
					print math.exp(-(min(diff, 0.0))**2)
				print 'nextone'
				humanstates = X[originalind][-1].human['B'].x
				pos = [x[1] for x in humanstates]
				for k in range(len(pos)-1):
					diff = pos[k+1] - pos[k]
					print 'diff', diff
					print math.exp(-(min(diff, 0.0))**2)
				print [truepick, truefeature]	
				mypick = select(X[originalind][-1])	
				mypick = 1 if mypick == 'A' else 0
				myfeature = int(input('Enter your number: '))
				print mypick
				
				if mypick == truepick:
					count += 1
				mypicks.append(mypick)
				myfeatures.append(myfeature)'''
				
				#oracle feature
				truefeature = np.argmax(np.absolute(W * df))
				
				#noisy feature
				pick = np.random.binomial(1, ppref[locations[ind_opt]])
				ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
				
				if noise == 0:
					p_phis = oracle(w, W, df, True, beta1_model, beta2_model, true_feature)
				elif noise == 1:
					p_phis = noisy(w, df, True, pick, beta1_model, beta2_model, ind_feature)
				
				#save oracle choice
				if np.dot(W, df) > 0:
					oracledata.append(['A', truefeature])
				else:
					oracledata.append(['B', truefeature])
			
				optindices.append(ind_opt)
				indices.append(locations[ind_opt])
				prob.append(p_w)
				
				
			filename = 'plfq_run' + r + '_noise' + noise + '_gt' + index + 'npy'
			np.save(filename, [prob, indices, oracledata])
			
			#################  PLOT RESULT ##########################################
							
			data = np.array([k[index] for k in prob])
			plt.plot(data, '#ffa500', linewidth = 3., label = 'PL_FQ')
			plt.legend(loc = 4)
			plt.xlabel('iterations')
			plt.ylabel(r'p($\theta_{GT}$)')
			plt.show()
			
#####################################   METHOD 2 #################################################
	
	if method == 2:
		print "\nRunning method 2: probabilistic comparison learning"
		
		for index in ind[0]:
		
			####################### INPUTS ############################################
			#temporary variables
			indices = []
			prob = []
			saveoracle = []
			
			#model parameters: rationality coefficients
			beta1_model = 2.
			beta1_sim = 2.
			
			#query pool, weight space and 
			cand_phis = train_phis
			locations = np.arange(len(train_phis))
			w = M.copy()
			
			###################### ALGORITHM #########################################
			
			p_phis = np.ones(w.shape[0])
			p_w = np.ones(w.shape[0])
			ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
			wdotx = np.dot(w,np.transpose(cand_phis))
			z1 = 1/(1+np.exp(-beta1_model * wdotx))
			z2 = 1/(1+np.exp(beta1_model * wdotx))
				
			for j in range(N):
				p_w = p_w * p_phis
				p_w = p_w/np.sum(p_w)
				obj = np.dot(np.transpose(p_w), z1) * np.dot(np.transpose(p_w), z2)
				df = cand_phis[np.argmax(obj)]
				ind_opt = np.argmax(obj)			
			
				if noise == 0:
					p_phis = oracle(w, W, df, False, beta1_model, 0., 0.)
				elif noise == 1:
					pick = np.random.binomial(1, ppref[locations[ind_opt]])
					p_phis = noisy(w, df, False, pick, beta1_model, 1., 1.)
				
				if np.dot(W, df) > 0:
					saveoracle.append(['A', np.argmax(np.absolute(W * df))])
				else:
					saveoracle.append(['B', np.argmax(np.absolute(W * df))])
	
				cand_phis = np.delete(cand_phis, ind_opt, 0)
				z1 = np.delete(z1, ind_opt, 1)
				z2 = np.delete(z2, ind_opt, 1)
				indices.append(locations[ind_opt])
				locations = np.delete(locations, ind_opt)
				prob.append(p_w)
		
			filename = 'pl_run' + r + '_noise' + noise + '_gt' + index + 'npy'
			np.save(filename, [prob, indices, oracledata])
			
			#################  PLOT RESULT ##########################################
			
			data = np.array([k[index] for k in prob])
			plt.plot(data, '#696969', linewidth = 3., label = 'PL', linestyle = '-.')
			plt.legend(loc = 4)
			plt.xlabel('iterations')
			plt.ylabel(r'p($\theta_{GT}$)')
			plt.show()
			
	
#####################################   METHOD 3 #################################################		
			
	if method == 3:
		print "\nRunning method 3: probabilistic comparison learning with feature query and query skip model"
		
		for index in ind[0]:
			
			####################### INPUTS ############################################
			
			#temporary variables
			oracledata = []
			optindices = []
			indices = []
			prob = []
			
			#model parameters: rationality coefficient and threshold probability
			beta1_model = 5.
			beta2_model = 2.5
			beta1_sim = 5.
			beta2_sim = 2.5
			eps_model = 0.06
			eps_sim = 0.06
		
			#query pool, weight space and true reward
			w = M.copy()
			cand_phis = train_phis
			W = M[index]
			
			###################### ALGORITHM #########################################
			
			p_phis = np.ones(w.shape[0])
			p_w = np.ones(w.shape[0])
			p_feature = np.zeros(w.shape)
			locations = np.arange(len(train_phis))
			ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
			pfeat = np.exp(beta2_sim * np.absolute(W*cand_phis))
			sumarray = np.sum(pfeat, axis = 1)
			for k in range(len(sumarray)):
				pfeat[k] = pfeat[k]/sumarray[k]
			wdotx = np.dot(w,np.transpose(cand_phis))
			z1 = 1/(1+np.exp(-beta1_model * wdotx))
			z2 = 1/(1+np.exp(beta1_model * wdotx))
		
			for j in range(N):
				if len(optindices) >= 1:
					data = np.delete(data, optindices[-1], 0)
					z1 = np.delete(z1, optindices[-1], 1)
					z2 = np.delete(z2, optindices[-1], 1)
					locations = np.delete(locations, optindices[-1])
				else:
					data = cand_phis
				
				p_w = p_w * p_phis
				p_w = p_w/np.sum(p_w)
				obj_baseline = np.zeros(len(data))
				obj_cirlf = np.zeros(len(data))
				
				
				for i in range(len(data)):
					x = data[i]
					den = np.sum(np.exp(beta2_model * np.absolute(w*x)), axis = 1)
					remA1 = 0.0
					remA2 = 0.0
					remB1 = 0.0
					remB2 = 0.0
					for n in range(w.shape[1]):
						num = np.exp(beta2_model * np.absolute(w[:,n]*x[n]))
						feat = num/den
						choiceA = z1[:, i] * feat
						choiceB = z2[:, i] * feat
						remA1 = remA1 + choiceA 
						remB1 = remB1 + choiceB
						remA2 = remA2 + 1. - choiceA 
						remB2 = remB2 + 1. - choiceB
						p_feature[:, n] = feat
					fmatlow = (p_feature > (1./len(W) - eps_model))
					fmathigh = (p_feature <= (1./len(W) + eps_model))
					vmat = np.dot(fmatlow * fmathigh, np.ones(w.shape[1]))
					delta = (vmat == w.shape[1])
					obj_baseline[i] = np.dot(delta * p_w, z1[:, i]) * np.dot(delta * p_w, z2[:, i])
					obj_cirlf[i] = np.dot((1 - delta) * p_w, remA1) * np.dot((1 - delta) * p_w, remA2) + np.dot((1 - delta) * p_w, remB1) * np.dot((1 - delta) * p_w, remB2)
				
				obj = obj_baseline + obj_cirlf
				df = data[np.argmax(obj)]
				ind_opt = np.argmax(obj)
			
				#simulate oracle choice
				if np.dot(W, df) > 0:
					oracledata.append(['A', np.argmax(np.absolute(W * df))])
				else:
					oracledata.append(['B', np.argmax(np.absolute(W * df))])
			
				#simulate noisy human choice
				pick = np.random.binomial(1, ppref[locations[ind_opt]])
				ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
			
				#no feature pick
				temp = np.exp(beta2_sim * np.abs(W*df))/np.sum(beta2_sim * np.exp(np.abs(W*df)))
				if ( temp > 1./len(W) - eps_sim).all() and (temp <= 1./len(W) + eps_sim).all():
					p_phis = noisy(w, df, False, pick, beta1_model, beta2_model, ind_feature)
				else:
					p_phis = noisy(w, df, True, pick, beta1_model, beta2_model, ind_feature)
					
				optindices.append(ind_opt)
				indices.append(locations[ind_opt])
				prob.append(p_w)
				
			filename = 'plfqskip_run' + r + '_noise' + noise + '_gt' + index + 'npy'
			np.save(filename, [prob, indices, oracledata])
			
			#################  PLOT RESULT ##########################################
			
			data = np.array([k[index] for k in prob])
			plt.plot(data, '#696969', linewidth = 3., label = 'PL_FQ_skip', linestyle = '-.')
			plt.legend(loc = 4)
			plt.xlabel('iterations')
			plt.ylabel(r'p($\theta_{GT}$)')
			plt.show()
			
			
	print "time elapsed: ", time.time() - start		
		
		
			


		
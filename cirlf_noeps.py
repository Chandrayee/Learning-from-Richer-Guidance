#!/usr/bin/env python

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
import pywren
import pickle


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
	
def increasedata(x, fn):
	x = list(x)
	moredata = np.load(fn)
	for i in range(len(moredata)):
		x.append(moredata[i])
	x = np.array(x)
	return x	
				
if __name__=='__main__':
	import logging
	start = time.time()
	train_phis = np.load('corrmdf.npy')
	train_phis = np.array([k/max(1, np.linalg.norm(k)) for k in train_phis])
	
	
	M = np.load('weights2000.npy')
	M = np.array([k/np.linalg.norm(k) for k in M])
	
	ind = [1759, 600, 12, 541, 7, 49, 9, 13, 4, 8, 704, 206, 165, 81, 1971, 902, 1950, 180, 822, 977]
	params = [0.01, 1.465, 5., 10., 30., 40.]
	index = 1759
	W = M[index]
	
		
	def models(run):
		beta1_model = 100.
		beta2_model = 2.5
		beta1_sim = 100.
		beta2_sim = 0.0001
		cand_phis = train_phis
		locations = np.arange(len(train_phis))
		w = M.copy()
		p_phis = np.ones(w.shape[0])
		p_w = np.ones(w.shape[0])
		ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
		pfeat = np.exp(beta2_sim * np.absolute(W*cand_phis))
		sumarray = np.sum(pfeat, axis = 1)
		for k in range(len(sumarray)):
			pfeat[k] = pfeat[k]/sumarray[k]	
		indices = []
		optindices = []
		prob = []
		oracledata = []

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
			
				
			#simulate oracle choice
			truefeature = np.argmax(np.absolute(W * df))
			'''ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
			p_phis = oracle(w, W, df, True, beta1_model, beta2_model, ind_feature)'''
			
			if np.dot(W, df) > 0:
				oracledata.append(['A', truefeature])
			else:
				oracledata.append(['B', truefeature])
	
			
			#simulate noisy human choice
			pick = np.random.binomial(1, ppref[locations[ind_opt]])
			ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
			p_phis = noisy(w, df, True, pick, beta1_model, beta2_model, ind_feature)
			
			optindices.append(ind_opt)
			indices.append(locations[ind_opt])
			prob.append(p_w)
		
		return [run, prob, indices, oracledata, p_phis, p_w]
		
	
	def models2(alldata):
		beta1_model = 100.
		beta2_model = 2.5
		beta1_sim = 100.
		beta2_sim = 0.0001
		cand_phis = train_phis
		locations = np.arange(len(train_phis))
		w = M.copy()
		ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
		pfeat = np.exp(beta2_sim * np.absolute(W*cand_phis))
		sumarray = np.sum(pfeat, axis = 1)
		for k in range(len(sumarray)):
			pfeat[k] = pfeat[k]/sumarray[k]	
		optindices = []
		prob = []
		oracledata = []
		wdotx = np.dot(w,np.transpose(cand_phis))
		z1 = 1/(1+np.exp(-beta1_model * wdotx))
		z2 = 1/(1+np.exp(beta1_model * wdotx))
		run = alldata[0]
		indices = alldata[1]
		p_phis = alldata[2]
		p_w = alldata[3]
		z1 = np.delete(z1, indices, 1)
		z2 = np.delete(z2, indices, 1)
		data = np.delete(cand_phis, indices, 0)
		locations = np.delete(locations, indices)
		
		for j in range(10):
			if len(optindices) >= 1:
				data = np.delete(data, optindices[-1], 0)
				z1 = np.delete(z1, optindices[-1], 1)
				z2 = np.delete(z2, optindices[-1], 1)
				locations = np.delete(locations, optindices[-1])
							
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
			
				
			#simulate oracle choice
			truefeature = np.argmax(np.absolute(W * df))
			'''ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
			p_phis = oracle(w, W, df, True, beta1_model, beta2_model, ind_feature)'''
			
			if np.dot(W, df) > 0:
				oracledata.append(['A', truefeature])
			else:
				oracledata.append(['B', truefeature])
			
			#simulate noisy human choice
			pick = np.random.binomial(1, ppref[locations[ind_opt]])
			ind_feature = np.where(np.random.multinomial(1, pfeat[locations[ind_opt]]) == 1)[0][0]
			p_phis = noisy(w, df, True, pick, beta1_model, beta2_model, ind_feature)
			
			optindices.append(ind_opt)
			indices.append(locations[ind_opt])
			prob.append(p_w)
		
		return [run, prob, indices, oracledata, p_phis, p_w]
		
	
	
	runs = list(np.arange(100))	
	filename = 'cirlfnoisy3_' + str(index) + 'bsim0001.pkl'
	
	'''wrenexec = pywren.default_executor()
	fut = wrenexec.map(models, runs)
	data = {}
	for f in fut:
		print f.callset_id
		res = f.result()
		data[str(res[0])] = {'prob': res[1], 'indices': res[2], 'oracle': res[3], 'p_phis':res[4], 'p_w': res[5]}
	with open(filename,'wb') as out:
		pickle.dump(data,out)'''
	
	with open(filename,'rb') as out:
		data = pickle.load(out)
		
	newarg = []
	for r in runs:
		newarg.append([r, data[str(r)]['indices'], data[str(r)]['p_phis'], data[str(r)]['p_w']])
		
	wrenexec = pywren.default_executor()
	fut = wrenexec.map(models2, newarg)
	data = {}
	for f in fut:
		print f.callset_id
		res = f.result()
		data[str(res[0])] = {'prob': res[1], 'indices': res[2], 'oracle': res[3], 'p_phis':res[4], 'p_w': res[5]}
	filename = 'cirlfnoisy4_' + str(index) + 'bsim0001.pkl'
	with open(filename,'wb') as out:
		pickle.dump(data,out)
		

	print time.time() - start
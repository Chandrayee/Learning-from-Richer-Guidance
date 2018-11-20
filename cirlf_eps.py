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
	
	

		 		
if __name__=='__main__':
	import logging
	start = time.time()
	train_phis = np.load('corrmdf.npy')
	train_phis = np.array([k/max(1, np.linalg.norm(k)) for k in train_phis])
	
	M = np.load('weights2000.npy')
	M = np.array([k/np.linalg.norm(k) for k in M])
	ind = [1759, 600, 12, 541, 7, 49, 9, 13, 4, 8, 704, 206, 165, 1971, 81, 902, 1950, 180, 822, 977]
	index = 704
	W = M[index]
	
	
	def models(run):
		beta1_model = 5.
		beta2_model = 2.5
		beta1_sim = 5.
		beta2_sim = 2.5
		eps_model = 0.06
		eps_sim = 0.06
		
		cand_phis = train_phis
		w = M.copy()
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
		
		oracledata = []
		optindices = []
		indices = []
		prob = []
		
		for j in range(8):
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
		
		return [run, prob, indices, oracledata, p_phis, p_w]
		
		
	def models2(alldata):
		beta1_model = 5.
		beta2_model = 2.5
		beta1_sim = 5. 
		beta2_sim = 2.5
		eps_model = 0.06
		eps_sim = 0.06
		
		cand_phis = train_phis
		w = M.copy()
		locations = np.arange(len(train_phis))
		p_feature = np.zeros(w.shape)
		
		ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
		pfeat = np.exp(beta2_sim * np.absolute(W*cand_phis))
		sumarray = np.sum(pfeat, axis = 1)
		for k in range(len(sumarray)):
			pfeat[k] = pfeat[k]/sumarray[k]
		
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
		
		oracledata = []
		optindices = []
		prob = []
		
		for j in range(8):
			if len(optindices) >= 1:
				data = np.delete(data, optindices[-1], 0)
				z1 = np.delete(z1, optindices[-1], 1)
				z2 = np.delete(z2, optindices[-1], 1)
				locations = np.delete(locations, optindices[-1])
				
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
		
		return [run, prob, indices, oracledata, p_phis, p_w]
		
		
	runs = list(np.arange(100))	
	filename = 'cirlfnoisy4_' + str(index) + '_esim06_emodel06.pkl'
	
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
	filename = 'cirlfnoisy5_' + str(index) + '_esim06_emodel06.pkl'
	with open(filename,'wb') as out:
		pickle.dump(data,out)
	 
	print time.time() - start
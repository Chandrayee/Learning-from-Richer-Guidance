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
	
	
	def models(index):
		W = M[index]
		beta1_model = 100.
		beta1_sim = 100.
		cand_phis = train_phis
		locations = np.arange(len(train_phis))
		w = M.copy()
		p_phis = np.ones(w.shape[0])
		p_w = np.ones(w.shape[0])
		indices = []
		prob = []
		saveoracle = []
		ppref = [k for k in 1/(1 + np.exp(beta1_sim * (-np.sum(W*cand_phis, axis = 1))))]
		wdotx = np.dot(w,np.transpose(cand_phis))
		z1 = 1/(1+np.exp(-beta1_model * wdotx))
		z2 = 1/(1+np.exp(beta1_model * wdotx))

		#iterations
		for j in range(20):
			p_w = p_w * p_phis
			p_w = p_w/np.sum(p_w)
			#obj = np.dot(np.transpose(p_w), (z1 * z2)**2)
			obj = np.dot(np.transpose(p_w), z1)  *  np.dot(np.transpose(p_w), z2)
			df = cand_phis[np.argmax(obj)]
			ind_opt = np.argmax(obj)

			#simulate oracle choice
			#p_phis = oracle(w, W, df, False)

			if np.dot(W, df) > 0:
				saveoracle.append(['A', np.argmax(np.absolute(W * df))])
			else:
				saveoracle.append(['B', np.argmax(np.absolute(W * df))])

			
			#oracle
			p_phis = oracle(w, W, df, False, beta1_model, 0., 0)
			
			#noisy human choice
			'''pick = np.random.binomial(1, ppref[locations[ind_opt]])
			p_phis = noisy(w, df, False, pick, beta1_model, 1., 1.)'''
	
			cand_phis = np.delete(cand_phis, ind_opt, 0)
			z1 = np.delete(z1, ind_opt, 1)
			z2 = np.delete(z2, ind_opt, 1)
			indices.append(locations[ind_opt])
			locations = np.delete(locations, ind_opt)
			prob.append(p_w)
		
		return [index, prob, indices, saveoracle]
		
		
	wrenexec = pywren.default_executor()
	ind = [1759, 600, 12, 541, 7, 49, 9, 13, 4, 8, 704, 206, 165, 1971, 81, 902, 1950, 180, 822, 977]
	fut = wrenexec.map(models, ind)
	
	data = {}
	for f in fut:
		print f.callset_id
		res = f.result()
		data[str(res[0])] = {'prob': res[1], 'indices': res[2], 'oracle': res[3]}
	filename = 'baselineor.pkl'
	with open(filename,'wb') as out:
		pickle.dump(data,out)
	 
	print time.time() - start
	
			


		
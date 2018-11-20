#!/usr/bin/env python

from world import *
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
import cPickle as pickle
import math
import random

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
	
def replaceA(xs, us):
	for i in [1,2,3,4,5]:
		world.human['A'].x[i] = tt.cast(xs[i-1], dtype = 'float32')
	for i in range(5):
		world.human['A'].u[i].set_value(us[i])
	
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
	
			
			
def instructions():
	print "We want to know how you want an autonomous car to drive."
	print "We will show you a pair of paths that a car can take for a given traffic situation and you will tell us which of the two paths would you prefer. "
	print "These paths are short sequences of maneuvers each less than 10 seconds."
	print "We will repeat this question for 20-30 different traffic conditions. The study will take up-to 30 minutes."
	print "You will also tell us which feature influenced your decision the most. Let's elaborate a bit more on that"
	print "The car trajectories are generated by varying different features of the car like speed, distance from road boundaries etc."
	print "More precisely, these features are: lanes, fences, roads, cars, speed, rightlane, reverse"
	print "Lanes = Whether the car is staying in any lane"
	print "Fences = Whether the car is staying within the road boundary"
	print "Roads = Whether the car is aligned with the road"
	print "Cars = Whether the car is close to or far away from surrounding cars"
	print "Speed = We will show you the actual speed number"
	print "Right lane = Whether the car prefers to stay in the right lane"
	print "Reverse = If the car is going in reverse direction."
	print "PLEASE DO NOT ASSUME ANY TRAFFIC THAT YOU CANNOT SEE"	
	
	
def load_cpickle_gc(fn):
    output = open(fn, 'rb')

    # disable garbage collector
    gc.disable()

    data = pickle.load(output)

    # enable garbage collector again
    gc.enable()
    output.close()
    return data
	
if __name__=='__main__':
	optlist, args = getopt.gnu_getopt(sys.argv, 'm:')
	opts = dict(optlist)
	method = int(opts.get('-m', 1))
	train_phis = np.load('corrmdf.npy')
	originaldata = train_phis
	M = np.load('weights2000.npy')
	M = list(M)
	M.append([0.25, -0.75, 0.5, -1., 0.5, -0.25, -0.5])
	M = np.array([k/np.linalg.norm(k) for k in M])
	beta1_model = 2.5
	beta2_model = 2.5
	beta1_sim = 2.5
	beta2_sim = 2.5
	fn = 'corrdata.pkl'
	with open(fn, 'rb') as out:
		X = pickle.load(out)
	train_phis = np.array([k/max(1, np.linalg.norm(k)) for k in train_phis])

	
	instructions()
	data = {'cirlf': [], 'baseline': []}
	
	if method == 1:
		print "Starting method 1"	
		cand_phis = train_phis
		w = M.copy()
		p_phis = np.ones(w.shape[0])
		p_w = np.ones(w.shape[0])
		locations = np.arange(len(train_phis))
		prob = []
		choice = []
		indices = []
		wdotx = np.dot(w,np.transpose(cand_phis))
		z1 = 1/(1+np.exp(-beta1_model * wdotx))
		z2 = 1/(1+np.exp(beta1_model * wdotx))
		savenorm = []
		count = 0
		for j in range(100):
			p_w = p_w * p_phis
			p_w = p_w/np.sum(p_w)
			obj = np.dot(np.transpose(p_w), z1) * np.dot(np.transpose(p_w), z2)
			df = cand_phis[np.argmax(obj)]
			ind_opt = np.argmax(obj)
			ind_orig = locations[ind_opt]
			print originaldata[ind_orig]
			
			#show case
			states = X[ind_orig][4]
			sA = round(np.linalg.norm(states.human['A'].x), 2)
			sB = round(np.linalg.norm(states.human['B'].x), 2)
			print [sA, sB]
			snorm = round(np.linalg.norm([sA,sB]), 1)
			print snorm
			if (sA == sB) or (snorm in savenorm):
				cand_phis = np.delete(cand_phis, ind_opt, 0)
				z1 = np.delete(z1, ind_opt, 1)
				z2 = np.delete(z2, ind_opt, 1)
				#indices.append(locations[ind_opt])
				locations = np.delete(locations, ind_opt)
				p_phis = np.ones(w.shape[0])
				print 'repeat'
				continue
			else:
				pick = select(states)
				savenorm.append(snorm)
				count += 1
		
			#human choice
			print 'good'
			if pick == 'A':
				pick = 1
				print "You chose A"
			elif pick == 'B':
				pick = 0
				print "You chose B"
			p_phis = noisy(w, df, False, pick, beta1_model, 0., 0.)
			prevA = sA
			prevB = sB
		
		
			cand_phis = np.delete(cand_phis, ind_opt, 0)
			z1 = np.delete(z1, ind_opt, 1)
			z2 = np.delete(z2, ind_opt, 1)
			indices.append(locations[ind_opt])
			locations = np.delete(locations, ind_opt)
			prob.append(p_w)
			choice.append(['A' if pick == 1 else 'B'])
			
			if count == 20:
				print 'count: ', count
				break
		baselineweight = M[np.argmax(prob[-1])]	
		baselineindex = np.argmax(prob[-1])
		data['baseline'].append([{'prob': prob}, {'choice': choice}, {'indices': indices}, {'w': baselineweight}, {'i': baselineindex}])
		
	method = 2
	if method == 2:
		print "Starting method 2"
		cand_phis = train_phis
		w = M.copy()
		p_phis = np.ones(w.shape[0])
		p_w = np.ones(w.shape[0])
		locations = np.arange(len(train_phis))
		prob = []
		choice = []
		indices = []
		wdotx = np.dot(w,np.transpose(cand_phis))
		z1 = 1/(1+np.exp(-beta1_model * wdotx))
		z2 = 1/(1+np.exp(beta1_model * wdotx))
		count = 0
		savenorm = []
		for j in range(100):
			p_w = p_w * p_phis
			p_w = p_w/np.sum(p_w)
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
					remA = remA + np.dot(p_w, choiceA) * np.dot(p_w, 1. - choiceA)
					remB = remB + np.dot(p_w, choiceB) * np.dot(p_w, 1. - choiceB)
				obj[i] = remA + remB
			df = cand_phis[np.argmax(obj)]
			ind_opt = np.argmax(obj)
			ind_orig = locations[ind_opt]
		
			#show case
			states = X[ind_orig][4]
			sA = round(np.linalg.norm(states.human['A'].x), 2)
			sB = round(np.linalg.norm(states.human['B'].x), 2)
			print [sA, sB]
			snorm = round(np.linalg.norm([sA,sB]), 2)
			print snorm
			if (sA == sB) or (snorm in savenorm):
				cand_phis = np.delete(cand_phis, ind_opt, 0)
				z1 = np.delete(z1, ind_opt, 1)
				z2 = np.delete(z2, ind_opt, 1)
				'''indices.append(locations[ind_opt])'''
				locations = np.delete(locations, ind_opt)
				p_phis = np.ones(w.shape[0])
				print 'repeat'
				continue
			else:
				pick = select(states)
				savenorm.append(snorm)
				count += 1
		
			#human choice
			print pick
			if pick == 'A':
				pick = 1
				print "You chose A"
			elif pick == 'B':
				pick = 0
				print "You chose B"
			elif pick == None:
				cand_phis = np.delete(cand_phis, ind_opt, 0)
				z1 = np.delete(z1, ind_opt, 1)
				z2 = np.delete(z2, ind_opt, 1)
				'''indices.append(locations[ind_opt])'''
				locations = np.delete(locations, ind_opt)
				p_phis = np.ones(w.shape[0])
				continue
			
			
			print "Features: "
			print "LANES: Following lane center - 0" 
			print "FENCES: Driving off the road - 1"
			print "ROAD: Alignment with road - 2"
			print "CARS: Collision/safe distance with cars - 3" 
			print "SPEED: Speed - 4"
			print "RIGHTLANE: Right lane preference - 5"
			print "REVERSE: Reverse - 6"
			ind_feature = int(input('Which of the above features of the driving behavior influenced your decision the most? Enter the number beside the feature: '))
			p_phis = noisy(w, df, True, pick, beta1_model, beta2_model, ind_feature)
		
				
			cand_phis = np.delete(cand_phis, ind_opt, 0)
			z1 = np.delete(z1, ind_opt, 1)
			z2 = np.delete(z2, ind_opt, 1)
			indices.append(locations[ind_opt])
			locations = np.delete(locations, ind_opt)
			prob.append(p_w)
			choice.append(['A' if pick == 1 else 'B', ind_feature])
			
			if count == 20:
				print 'count: ', count
				break
		
		cirlfweight = M[np.argmax(prob[-1])]	
		cirlfindex = np.argmax(prob[-1])
		data['cirlf'].append([{'prob': prob}, {'choice': choice}, {'indices': indices}, {'w': cirlfweight}, {'i': cirlfindex}])
		
		fn = 'user_me.pkl'
		with open(fn, 'wb') as out:
			pickle.dump(data, out)
	
		print np.dot(cirlfweight, baselineweight)
		print baselineweight
		print cirlfweight
		if np.dot(cirlfweight, baselineweight) >= 1.:
			print 'end study'
		else:
			method = 3
			
		print cirlfindex
		print baselineindex
				
	if method == 3:
		print "Starting method 3"
		countA = 0
		countB = 0
		#with open('comparisonrewardsdata.pkl', 'r') as out:
		fn = 'userstudysnaps30.npy'
		data = np.load(fn)
		count = 0
		for n in range(len(data)):
			snapA = data[n][cirlfindex]
			snapB = data[n][baselineindex]
			xs = snapA.human['A'].x
			us = snapA.human['A'].u
			newsnap = TrajectorySnapshot(xs,us)
			snapB.human['A'] = newsnap
			valA = np.linalg.norm(snapB.human['A'].x)
			valB = np.linalg.norm(snapB.human['B'].x)
			if count == 10:
				break
			if np.abs(valA - valB) < 0.1:
				print 'count: ', count
				continue
			else:
				pick = select(snapB)
				count += 1
			
			
			if pick == 'A':
				countA += 1
			elif pick == 'B':
				countB += 1
				
		
		print [countA, countB]
		
		if countA >= countB:
			print "Cirlf gave the true weight"
			print "True reward function is: ", cirlfweight
		else:
			print "Baseline gave the true weight"
			print "True reward function is: ", baselineweight
			
		
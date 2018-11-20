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
import pickle
import math
import random
import gc


def load_cpickle_gc(fn):
    output = open(fn, 'rb')

    # disable garbage collector
    gc.disable()

    data = pickle.load(output)

    # enable garbage collector again
    gc.enable()
    output.close()
    return data
    
fn = 'userstudysnaps.pkl'
#data = load_cpickle_gc(fn)
with open(fn, 'r') as out:
	data = pickle.load(out)
print 'unpickle done'
	
snap = []
for i in range(len(data)):
	snap.append(data[i])
	
np.save('userstudysnaps.npy',snap)
    
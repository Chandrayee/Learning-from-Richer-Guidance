from world import *
from visual import select
import numpy as np

data = np.load('showdata.npy')
for i in range(len(data)):
	select(data[i])
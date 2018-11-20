from world import *
import numpy as np
import math
import pickle

M = np.load('weights2000.npy')
M = np.array([k/np.linalg.norm(k) for k in M])
fn = 'testdata.pkl'
with open(fn, 'rb') as out:
	envs = pickle.load(out)
data = []
envcount = 0

for env in envs[:10]:
	envcount += 1
	xpos = [0.0, env[0][0]]
	ypos = [env[0][1] + 0.2, env[0][1] - 0.2]
	pos = [[x,y] for (x,y) in zip(xpos, ypos)]
	angle = [math.pi/2.5, math.pi/2.]
	speed = [0.6]
	s = []
	[s.append([x[0],x[1], y, z]) for x in pos for y in angle for z in speed]
	print len(s)
	print s
	for x0 in s:
		dat = {'env': env, 'x0': x0, 'weight': [], 'snap': [], 'fvector': [], 'mdf': []}
		#cirlf
		count = 0
		for weight in M:
			count += 1
			world.w.set_value(weight) 
			#optimization
			for x, B in iter(world.bounds.iteritems()):
				x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
			world.robots[0].x[0].set_value(env[0])
			for i in [1,2,3,4,5]:
				world.robots[0].x[i] = tt.cast(env[i], dtype = 'float32')
			world.x0.set_value(x0)	
			for opt in world.human_optimizer.values():
				#world.w.set_value([np.random.normal() for _ in features])
				opt.maximize(bounds=world.bounds)
			snap = world.dump()
			dat['weight'].append(weight)
			dat['snap'].append(snap)
			dat['mdf'].append(world.mdf)
			dat['fvector'].append(world.f1)
		print count
		data.append(dat)
print envcount
		
with open('rewardsdata_all2.pkl', 'w') as out:
	pickle.dump(data,out)
			
			
		
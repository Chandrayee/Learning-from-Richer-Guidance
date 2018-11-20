from world import *
from visual import select
import sys
import getopt


if __name__=='__main__':
	optlist, args = getopt.gnu_getopt(sys.argv, 's:')
	opts = dict(optlist)
	sindex = int(opts.get('-s', 7))
	print 'start ...............................'
	#get environments
	fn = 'trainingdata.pkl'
	with open(fn, 'rb') as out:
		envs = pickle.load(out)
	env = envs[2]

	#generate start states
	xpos = [-0.13, 0., 0.13] 
	ypos = [0., 0., 0.]
	if env[0][0] < -0.065:
		ypos[0] = env[0][1] + 0.25
	elif env[0][0] > 0.065:
		ypos[2] = env[0][1] - 0.2
	else:
		ypos[1] = env[0][1] + 0.15
	pos = [[x,y] for (x,y) in zip(xpos, ypos)]
	angle = [math.pi/2., math.pi/2.5, math.pi - math.pi/2.5]
	speed = [0.5, 0.8]
	s = []
	[s.append([x[0],x[1], y, z]) for x in pos for y in angle for z in speed] 
	#get correct weight
	M = np.load('weights2000.npy')
	M = np.array([k/np.linalg.norm(k) for k in M])
	ind = [0, 4, 5, 6, 7, 8, 9, 11, 12, 13, 1500, 1773, 816, 1589, 541, 51, 600, 49, 1759]
	
	data = []
	count = 0
	for x0 in [s[sindex]]: 
		print 'start is: ', sindex
		for k in range(len(ind)):
			weight = M[ind[k]]
			print 'index A: ', ind[k]
			#optimization
			for x, B in iter(world.bounds.iteritems()):
				x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
			world.robots[0].x[0].set_value(env[0])
			for i in [1,2,3,4,5]:
				world.robots[0].x[i] = tt.cast(env[i], dtype = 'float32')
			world.x0.set_value(x0)
			world.w.set_value(weight)	
			for opt in world.human_optimizer.values():
				#world.w.set_value([np.random.normal() for _ in features])
				opt.maximize(bounds=world.bounds)
			#world.new_optimizer.maximize(bounds=world.bounds)
			print np.dot(world.w.get_value(), world.mdf)
			xs = []
			us = []
			for i in [1, 2, 3, 4, 5]:
				xs.append(world.human['A'].x[i].eval())
			for i in range(5):
				us.append(world.human['A'].u[i].get_value())
			print 'traj for 0'
			#select(world.dump())

			prevreward = 0.
			rewards = []
			for n in ind[k+1:]:
				print 'index B: ', n
				weight = M[n]
				#optimization
				for x, B in iter(world.bounds.iteritems()):
					x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
				world.robots[0].x[0].set_value(env[0])
				for i in [1,2,3,4,5]:
					world.robots[0].x[i] = tt.cast(env[i], dtype = 'float32')
				world.x0.set_value(x0)
				world.w.set_value(weight)	
				for opt in world.human_optimizer.values():
					#world.w.set_value([np.random.normal() for _ in features])
					opt.maximize(bounds=world.bounds)
				#world.new_optimizer.maximize(bounds=world.bounds)
				for i in [1,2,3,4,5]:
					world.human['A'].x[i] = tt.cast(xs[i-1], dtype = 'float32')
				for i in range(5):
					world.human['A'].u[i].set_value(us[i])
				x = world.fvector('A').eval()
				y = world.f2
				dat = world.mdf
				reward = np.dot(M[0], dat)
				print reward
				if (reward not in rewards) and (np.abs(reward) > 0.2) and (np.abs(reward - prevreward) > 0.2):
					snap = world.dump()
					print 'trajsaved: '
					#select(snap)
					data.append([{'env': env},{'x0': x0},{'w': [M[ind[k]],M[n]]},{'mdf': dat}, snap])
					count += 1
					print 'count: ', count
				rewards.append(reward)
				prevreward = reward

	if count > 0:
		print count
		fn = 'newdata2_' + str(sindex) + '.pkl'
		with open(fn,'wb') as out:
			pickle.dump(data,out)


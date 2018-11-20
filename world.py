import numpy as np
from utils import vector, matrix, Maximizer
from dynamics import CarDynamics
from scipy.interpolate import interp1d
import pickle
import math
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
import itertools
from visual import select

def dumpable(obj):
    if isinstance(obj, Dumpable):
        return True
    if isinstance(obj, list) or isinstance(obj, tuple):
        return all([dumpable(x) for x in obj])
    if isinstance(obj, dict):
        return all([dumpable(x) for x in obj.values()])
    return False

def dump(obj):
    if isinstance(obj, list):
        return [dump(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple([dump(x) for x in obj])
    elif isinstance(obj, dict):
        return {k: dump(v) for k, v in iter(obj.items())}
    elif isinstance(obj, Dumpable):
        return obj.dump()
    else:
        ret = Snapshot()
        for k, v in iter(vars(obj).items()):
            if dumpable(v):
                setattr(ret, k, dump(v))
        return ret

def load(obj, snapshot):
    if isinstance(snapshot, list) or isinstance(snapshot, tuple):
        if len(obj)!=len(snapshot):
            raise Exception('Length mistmatch.')
        for x, y in zip(obj, snapshot):
            load(x, y)
    elif isinstance(snapshot, dict):
        for k, v in snapshot.iteritems():
            load(obj[k], v)
    elif isinstance(obj, Dumpable):
        obj.load(snapshot)
    else:
        for k, v in vars(snapshot).iteritems():
            if not hasattr(obj, k):
                continue
            load(getattr(obj, k), v)

class Snapshot(object):
    @property
    def answer(self):
        if not hasattr(self, '_answer'):
            self.answer = None
        return self._answer
    @answer.setter
    def answer(self, value):
        self._answer = value
    @property
    def user(self):
        if not hasattr(self, '_user'):
            self.user = None
        return self._user
    @user.setter
    def user(self, value):
        self._user = value
    def view(self, key):
        ret = Snapshot()
        for k, v in iter(vars(self).items()):
            setattr(ret, k, v[key] if isinstance(v, dict) else v)
        return ret
    def keys(self):
        ret = set()
        for k, v in iter(vars(self).items()):
            if isinstance(v, dict):
                ret = ret|set(v.keys())
        return list(sorted(ret))
    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self, f)
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class Dumpable(object): pass

class Lane(Dumpable):
    def __init__(self, p, q, w):
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q-self.p)/np.linalg.norm(self.q-self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])
    def gaussian(self, x, width=0.5):
        d = (x[0]-self.p[0])*self.n[0]+(x[1]-self.p[1])*self.n[1]
        return tt.exp(-0.5*d**2/(width**2*self.w*self.w/4.))
    def direction(self, x):
        return tt.cos(x[2])*self.m[0]+tt.sin(x[2])*self.m[1]
    def shifted(self, m):
        return Lane(self.p+self.n*self.w*m, self.q+self.n*self.w*m, self.w)
    def dump(self):
        return self
    def load(self, snapshot):
        self.__init__(snapshot.p, self.q, self.w)

class Trajectory(Dumpable):
    def __init__(self, T, dyn, x0=None):
        self.x = [vector(4) if x0 is None else x0]
        self.u = [vector(2) for _ in range(T)]
        self.dyn = dyn
        for t in range(T):
            self.x.append(dyn(self.x[t], self.u[t]))
    def dump(self, preset = True):
        us = [u.get_value() for u in self.u]
    	if preset:
    		xs = []
        	for i in range(len(self.x)):
        		xs.append(self.x[i].eval())
    	else: 
    		xs = [self.x[0].get_value()]
        	for t in range(len(self.u)):
        		xs.append(self.dyn.compiled(xs[t], us[t]))
        return TrajectorySnapshot(xs, us)
    def load(self, snapshot):
        if len(snapshot.x)!=len(self.x) or len(snapshot.u)!=len(self.u):
            raise Exception('Trajectory length mismatch.')
        for x, u, y in zip(snapshot.x[:-1], snapshot.u, snapshot.x[1:]):
            if sum(np.abs(y-self.dyn.compiled(x, u)))>1e-5:
                raise Exception('Dynamics mistmatch.')
        self.x[0].set_value(snapshot.x[0])
        for u, v in zip(self.u, snapshot.u):
            u.set_value(v)

class TrajectorySnapshot(object):
    def __init__(self, xs, us):
        self.x = [np.asarray(x) for x in xs]
        self.u = [np.asarray(u) for u in us]
        self.ix = interp1d(np.asarray(range(self.T+1)), np.asarray(self.x), axis=0, kind='cubic')
        self.iu = interp1d(np.asarray(range(self.T)), np.asarray(self.u), axis=0, kind='cubic')
    def __getstate__(self):
        return (self.x, self.u)
    def __setstate__(self, state):
        xs, us = state
        self.__init__(xs, us)
    @property
    def T(self):
        return len(self.u)

class Feature(object):
    def __init__(self, f, name):
        self.f = f
        self.name = name
    def total(self, world, *args, **vargs):
        return sum(self.f(world.moment(t, *args, **vargs))
                for t in range(1, world.T+1))

def feature(f):
    return Feature(f, f.__name__)

@feature
def lanes(moment):
    return sum(lane.gaussian(moment.human) for lane in moment.lanes)

@feature
def fences(moment):
    return sum(fence.gaussian(moment.human) for fence in moment.fences)

@feature
def roads(moment):
    return sum(road.direction(moment.human) for road in moment.roads)

@feature
def speed(moment):
    #return (moment.human[3]-1.)**2
    return tt.exp(-0.5*((moment.human[3]-1.)/0.1)**2)

def car_gaussian(x, y, height=.07, width=.03):
    d = (x[0]-y[0], x[1]-y[1])
    dh = tt.cos(x[2])*d[0]+tt.sin(x[2])*d[1]
    dw = -tt.sin(x[2])*d[0]+tt.cos(x[2])*d[1]
    return tt.exp(-0.5*(dh*dh/(height*height)+dw*dw/(width*width)))

@feature
def cars(moment):
    return sum(car_gaussian(robot, moment.human) for robot in moment.robots)
    
#change here    
@feature
def rightlane(moment):
    return moment.lanes[2].gaussian(moment.human)
    
def smoothness(x, y):
	return (x[3] - y[3])**2
	
@feature
def smooth(moment):
    return smoothness(moment.human, moment.prev)
    
@feature
def reverse(moment):
		return tt.exp(-(tt.clip((moment.human[1] - moment.prev[1]), -10000., 0.0))**2)
	
@feature
def lanechange(moment):
	return np.abs((moment.human[2]-math.pi*0.5)*moment.human[3])
	    
#features = [lanes, fences, roads, cars, speed, rightlane, smooth, reverse, lanechange]
features = [lanes, fences, roads, cars, speed, rightlane, reverse]

class Moment(object):
    def __init__(self, world, t, version):
        self.lanes = world.lanes
        self.fences = world.fences
        self.roads = world.roads
        self.robots = [robot.x[t] for robot in world.robots]
        self.human = world.human[version].x[t]
        self.prev = world.human[version].x[t-1]

class World(object):
    def __init__(self, T, dyn, lanes=[], fences=[], roads=[], ncars=1):
        self.lanes = lanes
        self.fences = fences
        self.roads = roads
        self.robots = [Trajectory(T, dyn) for _ in range(ncars)]
        self.x0 = vector(4)
        self.human = {
            'A': Trajectory(T, dyn, self.x0),
            'B': Trajectory(T, dyn, self.x0)
        }
        #ground truth can be + or - one of the above arrays
        self.bounds = {}
        uB = [(-2., 2.), (-1., 1.)]
        xB = [(-0.15, 0.15), (-0.1, 0.2), (math.pi*0.4, math.pi*0.6), (0., 1.)]
        for robot in self.robots:
            for u in robot.u:
                self.bounds[u] = uB
            self.bounds[robot.x[0]] = xB
        for human in self.human.values():
            for u in human.u:
                self.bounds[u] = uB
            self.bounds[human.x[0]] = xB

        self.w = vector(len(features))
        self.samples = matrix(0, len(features))
        self.M = np.load('weights2000.npy')
    @property
    def df(self):
        if not hasattr(self, '_df'):
            self._df = th.function([], self.fvector('A')-self.fvector('B'))
        return self._df()
    @property
    def f1(self):
        if not hasattr(self, '_f1'):
            self._f1 = th.function([], self.fvector('A'))
        return self._f1()
    @property
    def f2(self):
        if not hasattr(self, '_f2'):
            self._f2 = th.function([], self.fvector('B'))
        return self._f2()
    @property
    def mdf(self):
        df = self.df
        mean = np.array([-0.0029, -0.0007, -0.001, 0.002, -0.0009, 0.01463, -0.00041])
        st = np.array([1., 0.48, 0.27, 0.42, 0.2, 0.8, 0.00124])
        return np.divide(df, st)
    @property
    def ndf(self):
        df = self.df
        return df/max(1, np.linalg.norm(df))
    @property
    def human_optimizer(self):
        if not hasattr(self, '_human_optimizer'):
            self._human_optimizer = {
                v: Maximizer(tt.dot(self.w, self.fvector(v)), self.human[v].u)
                for v in self.human
            }
        return self._human_optimizer
    @property
    def new_optimizer(self):
    	if not hasattr(self, '_optimizer'):
    		df = self.fvector('A')-self.fvector('B')
    		st = np.array([1., 0.48, 0.27, 0.42, 0.2, 0.8, 0.00124])
    		phi = df/st
    		#phi = z/(1+tn.relu(z.norm(2)-1))
    		obj = tt.dot(self.w, phi)
    		variables = []
    		for human in self.human.values():
    			variables += human.u
    		self._optimizer = Maximizer(obj, variables)
    	return self._optimizer
    @property
    def optimizer(self):
        if not hasattr(self, '_optimizer'):
            df = self.fvector('A')-self.fvector('B')
            phi = df/(1+tn.relu(df.norm(2)-1))
            y = tt.dot(self.samples, phi)
            p = tt.sum(tt.switch(y<0, 1., 0.))
            q = tt.sum(tt.switch(y>0, 1., 0.))
            if not hasattr(self, 'avg_case'):
                obj = tt.minimum(tt.sum(1.-tt.exp(-tn.relu(y))), tt.sum(1.-tt.exp(-tn.relu(-y))))
            else:
                obj = p*tt.sum(1.-tt.exp(-tn.relu(y)))+q*tt.sum(1.-tt.exp(-tn.relu(-y)))
            #changes here
            #variables = [self.x0]
            variables = []
            for robot in self.robots:
            	#variables += [robot.x[0]]+robot.u
                variables += robot.u
            for human in self.human.values():
                variables += human.u
            self._optimizer = Maximizer(obj, variables)
        return self._optimizer
    def fvector(self, *args, **vargs):
        return tt.stacklists([f.total(self, *args, **vargs) for f in features])
    @property
    def T(self):
        return len(list(self.human.values())[0].u)
    def dump(self):
        return dump(self)
    def load(self, snapshot):
        load(self, snapshot)
    def moment(self, t, version):
        return Moment(self, t, version)        
    def randomize(self):
        for x, B in self.bounds.iteritems():
            x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B])) 
    def newgen(self):
    	data = []
    	for x, B in iter(self.bounds.iteritems()):
    		x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
    	for j in range(10):
    		self.x0.set_value(np.array([np.random.uniform(a, b) for (a, b) in self.bounds[self.human['A'].x[0]]]))
    		for human in self.human.values():
    			for u in human.u:
    				u.set_value(np.array([np.random.uniform(a, b) for (a, b) in self.bounds[u]]))
    		#select(self.dump())
    		data.append(self.ndf)
    	return data
    def optimalgen(self, indices, env, x0, weight = []):
    	data = []
    	for x, B in iter(self.bounds.iteritems()):
    		x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
    	self.robots[0].x[0].set_value(env[0])
    	for i in [1,2,3,4,5]:
    		self.robots[0].x[i] = tt.cast(env[i], dtype = 'float32')
    	self.x0.set_value(x0)		
    	if weight == []:
    		for index in indices:
    			self.w.set_value(self.M[index])
    			self.new_optimizer.maximize(bounds=self.bounds)
    			data.append(self.mdf)
    	else:
    		self.w.set_value(weight)
    		self.new_optimizer.maximize(bounds=self.bounds)
    		data.append(self.mdf)
    	return data
    def gen(self, samples, dumb=False, random_init=False):
    	self.samples.set_value(samples)
    	for x, B in iter(self.bounds.iteritems()):
    		x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
    	h = self.x0.get_value()[0:2]
     	r = self.robots[0].x[0].get_value()[0:2]
     	while np.linalg.norm(h-r) <= 0.13:
     		self.x0.set_value(np.array([np.random.uniform(a, b) for (a, b) in self.bounds[self.human['A'].x[0]]]))
     		self.robots[0].x[0].set_value([np.random.uniform(a, b) for (a, b) in self.bounds[self.robots[0].x[0]]])
     		h = self.x0.get_value()[0:2]
     		r = self.robots[0].x[0].get_value()[0:2]
    	if not random_init:
    		for opt in self.human_optimizer.values():
    			if len(samples)==0:
    				self.w.set_value(np.random.normal(size=len(features)))
    			else:
    				self.w.set_value(samples[np.random.choice(len(samples)), :])
    				#self.w.set_value(self.W)
    			opt.maximize(bounds=self.bounds)
    	if not dumb:
    		self.optimizer.maximize(bounds=self.bounds)
        

lane = Lane([0., -1.], [0., 1.], 0.13)
road = Lane([0., -1.], [0., 1.], 0.13*3)
env1 = {
    'lanes': [
        lane.shifted(0),
        lane.shifted(-1),
        lane.shifted(1)
    ],
    'fences': [
        lane.shifted(2),
        lane.shifted(-2)
    ],
    'roads': [
        road
    ]
}

world = World(5, CarDynamics(), **env1)
for v in ['A', 'B']:
    traj = world.human[v]
    traj.x[0].set_value([0., 0., math.pi/2., .5])
    for u in traj.u:
        u.set_value([0., 1.])
world.robots[0].x[0].set_value([0.13, 0., math.pi/2., 0.5])

#example1 = world.dump()

if __name__=='__main__':
	from visual import select
	#get environments
	fn = 'trainingdata.pkl'
	with open(fn, 'rb') as out:
		envs = pickle.load(out)
	env = envs[0]
	
	#generate start states
	xpos = [-0.13, 0., 0.13] 
	ypos = [0., 0., 0.]
	if env[0][0] < -0.065:
		ypos[0] = env[0][1] + 0.25
	elif env[0][0] > 0.065:
		ypos[2] = env[0][1] + 0.15
	else:
		ypos[1] = env[0][1] + 0.15
	pos = [[x,y] for (x,y) in zip(xpos, ypos)]
	angle = [math.pi/2., math.pi/2.5, math.pi - math.pi/2.5]
	speed = [0.5, 0.8]
	s = []
	[s.append([x[0],x[1], y, z]) for x in pos for y in angle for z in speed]
	x0 = s[0] 
	
	#get correct weight
	M = np.load('weights2000.npy')
	M = np.array([k/np.linalg.norm(k) for k in M])
	ind = [0, 4, 5, 6, 7, 8, 9, 11, 12, 13, 1500, 1773, 816, 1589, 541, 51, 600, 49, 1759]
	weight = M[4]
	
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
	select(world.dump())
	xs = []
	us = []
	for i in [1, 2, 3, 4, 5]:
		xs.append(world.human['B'].x[i].eval())
	for i in range(5):
		us.append(world.human['B'].u[i].get_value())
	
	
	weight = M[0]
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
	print np.dot(M[0], world.mdf)
	select(world.dump())
	for i in [1,2,3,4,5]:
		world.human['B'].x[i] = tt.cast(xs[i-1], dtype = 'float32')
	for i in range(5):
		world.human['B'].u[i].set_value(us[i])
	x = world.f1
	y = world.fvector('B').eval()
	print np.dot(M[0], world.mdf)
	select(world.dump())
	quit()

	

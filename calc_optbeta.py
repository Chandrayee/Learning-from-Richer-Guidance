from world import *
import numpy as np
from visual import select
import matplotlib.pyplot as plt


M = np.load('weights2000.npy')
M = list(M)
M.append([0.25, -0.75, 0.5, -1., 0.5, -0.25, -0.5])
M = np.array([k/np.linalg.norm(k) for k in M])
#W = M[1419]


'''data = np.load('cirlfoptimal.npy')
df = np.array(data[3])
mypick = np.array(data[4])
oracle = np.array(data[2])
myfeature = np.array(data[5])	
x = [1 if k[0]=='A' else -1 for k in oracle]
mypick = [1 if k == 1 else -1 for k in mypick]'''



train_phis = np.load('corrmdf.npy')
originaldata = train_phis
norms = []
names = ['swadha', 'jason', 'maryam','kai', '60949', '64942']
for n in names:
	fn = 'user_' + n + '.pkl'
	with open(fn, 'r') as out:
		data = pickle.load(out)

	ind = data['cirlf'][0][2]['indices']
	normdata = [np.linalg.norm(k) for k in originaldata[ind]]
	for i in range(len(ind)):
		norms.append(normdata[i])
ax = plt.subplot(111)
plt.hist(norms, bins = 20, edgecolor = 'grey', color = '#FFA500', alpha = 0.5)
plt.xlim(0,18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('cirlf_query.png')
plt.show()

norms = []
names = ['swadha', 'jason', 'maryam','kai']
for n in names:
	fn = 'user_' + n + '.pkl'
	with open(fn, 'r') as out:
		data = pickle.load(out)

	ind = data['baseline'][0][2]['indices']
	normdata = [np.linalg.norm(k) for k in originaldata[ind]]
	for i in range(len(ind)):
		norms.append(normdata[i])
ax = plt.subplot(111)
plt.xlim(0,18)
ax.hist(norms, bins = 20, edgecolor = 'grey', color = '#505050', alpha = 0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('baseline_query.png')
plt.show()

'''train_phis = np.array([k/max(1, np.linalg.norm(k)) for k in train_phis])
fn = 'user_swadha.pkl'
with open(fn, 'r') as out:
	data = pickle.load(out)
ind = data['cirlf'][0][2]['indices']
df = train_phis[ind]
W = data['cirlf'][0][3]['w']
mypick = np.array(data['cirlf'][0][1]['choice'])[:,0]
mypick = [1 if k == 'A' else -1 for k in mypick]
myfeature = np.array(data['cirlf'][0][1]['choice'])[:,1]
myfeature = [int(k) for k in myfeature]
plt.hist([np.linalg.norm(k) for k in originaldata[ind]])
plt.show()
ind = data['baseline'][0][2]['indices']
df = train_phis[ind]
mypick = np.array(data['baseline'][0][1]['choice'])[:,0]
mypick = [1 if k == 'A' else -1 for k in mypick]
plt.hist([np.linalg.norm(k) for k in originaldata[ind]])
plt.show()


points = np.arange(0.1,100, 0.1)
obj = []

for beta1 in points:
	f = 0
	for i in range(len(df)):
		f = f + np.log(1/(1 + np.exp(-mypick[i]*beta1*np.dot(W, df[i]))))
	obj.append(f)
	
plt.semilogx(points, obj)
plt.show()
beta1 = np.argmax(obj)
print points[beta1]

obj = []
for beta2 in points:
	f = 0
	for i in range(len(df)):
		feat = np.abs(np.exp(beta2*W*df[i]))
		f = f + np.log(1/(1 + np.exp(-mypick[i]*beta1*np.dot(W, df[i])))) + np.log(feat[myfeature[i]]/np.sum(feat))
	obj.append(f)

plt.semilogx(points, obj)
plt.show()	
beta2 = np.argmax(obj)	
print points[beta2]'''



def plottruereward(index):
	trueR = []
	for i in range(len(data)):
		w = data[i]['weight'][index]
		traj = data[i]['fvector'][index]
		trueR.append(np.dot(w,traj))
	trueR = np.mean(trueR)
	plt.plot(np.arange(40), trueR * np.ones(40), '--k', linewidth = 2., label = 'true reward')
	
def plotoraclereward(index, res, colour):
	maxindices = [np.argmax(k) for k in res[str(index)]['prob']]
	meanreward = []
	errreward = []
	for m in maxindices:
		rewardperiter = []
		for i in range(len(data)):
			w = data[i]['weight'][index]
			traj = data[i]['fvector'][m]
			rewardperiter.append(np.dot(w,traj))
		meanreward.append(np.mean(rewardperiter))
		errreward.append(np.std(rewardperiter)/np.sqrt(len(data)))
	plt.plot(np.arange(40), meanreward, colour, linewidth = 3., label = label)
	plt.errorbar(np.arange(40), meanreward, yerr=errreward, fmt = 'o', linewidth = 1., color = colour, markeredgewidth = 0.0)
	
	
def getrewardperrun(index, res, run):
	maxindices = [np.argmax(k) for k in res[str(run)]['prob']]
	meanreward = []
	errreward = []
	for m in maxindices:
		rewardperiter = []
		for i in range(len(data)):
			w = data[i]['weight'][index]
			traj = data[i]['fvector'][m]
			rewardperiter.append(np.dot(w,traj))
		meanreward.append(np.mean(rewardperiter))
		errreward.append(np.std(rewardperiter)/np.sqrt(len(data)))
	return meanreward
	
def plotnoisyreward(index, res, colour):
	runreward = []
	for run in range(len(res)):
		runreward.append(getrewardperrun(index, res, run))
		
	runreward = np.transpose(np.array(runreward))
	meanreward = np.mean(runreward, axis = 1)
	errreward = np.std(runreward, axis = 1)/np.sqrt(len(res))
	plt.plot(np.arange(40), meanreward, colour, linewidth = 3., label = label)
	plt.errorbar(np.arange(40), meanreward, yerr=errreward, fmt = 'o', linewidth = 1., color = colour, markeredgewidth = 0.0)
	
		
def initialize(fn):
	with open(fn, 'rb') as out:
		data = pickle.load(out)
	return data

index = 615

datafile = 'rewardsdata.pkl'
with open(datafile, 'rb') as out:
	data = pickle.load(out)
	
fn = 'baselineordata2' + str(index) + '.pkl'
res = initialize(fn)
plotorreward(index, res, '#696969')

fn = 'cirlfordata2' + str(index) + '.pkl'
res = initialize(fn)
plotorreward(index, res, 'red')
	
fn = 'baselinenoisy_' + str(index) + '.pkl'
res = initialize(fn)
plotnoisyreward(index, res, '#696969')

fn = 'cirlfnoisy_' + str(index) + '.pkl'
res = initialize(fn)
plotnoisyreward(index, res, 'red')

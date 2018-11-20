import pickle
import numpy as np


d = 2

filename = 'cirlfor_1.pkl'
with open(filename, 'rb') as out:
	data = pickle.load(out)
	
filename = 'cirlfor_2.pkl'
with open(filename, 'rb') as out:
	data1 = pickle.load(out)
	
'''filename = 'cirlfordata' + str(d) + '_3.pkl'
with open(filename, 'rb') as out:
	data2 = pickle.load(out)
	
filename = 'cirlfordata' + str(d) + '_4.pkl'
with open(filename, 'rb') as out:
	data3 = pickle.load(out)'''
	


ind = [1759, 600, 12, 541, 7, 49, 9, 13, 4, 8, 704, 206, 165, 1971, 81, 902, 1950, 180, 822, 977]

for index in ind:
	data[str(index)]['indices'] = data1[str(index)]['indices']

	for i in range(10):
		data[str(index)]['oracle'].append(data1[str(index)]['oracle'][i])
		data[str(index)]['prob'].append(data1[str(index)]['prob'][i])
		
	'''for i in range(10):
		data[str(index)]['oracle'].append(data2[str(index)]['oracle'][i])
		data[str(index)]['prob'].append(data2[str(index)]['prob'][i])
		
	for i in range(10):
		data[str(index)]['oracle'].append(data3[str(index)]['oracle'][i])
		data[str(index)]['prob'].append(data3[str(index)]['prob'][i])'''
		
		
filename = 'cirlfor.pkl'

with open(filename, 'wb') as out:
	pickle.dump(data, out)
		
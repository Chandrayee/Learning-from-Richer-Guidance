import pickle
import numpy as np

ind = [1759, 600, 12, 541, 7, 49, 9, 13, 4, 8, 704, 206, 165, 81, 1971, 902, 1950, 180, 822, 977]
#1759, 4, 165, 600, 9, 8, 704, 1971, 81 (12, 7)
index = 1759
print index

filename = 'cirlfnoisy1_' + str(index) + 'bsim0001.pkl'
with open(filename, 'rb') as out:
	data = pickle.load(out)
	
filename = 'cirlfnoisy2_' + str(index) + 'bsim0001.pkl'
with open(filename, 'rb') as out:
	data1 = pickle.load(out)
	
filename = 'cirlfnoisy3_' + str(index) + 'bsim0001.pkl'
with open(filename, 'rb') as out:
	data2 = pickle.load(out)
	
filename = 'cirlfnoisy4_' + str(index) + 'bsim0001.pkl'
with open(filename, 'rb') as out:
	data3 = pickle.load(out)
	
'''filename = 'cirlfnoisy5_' + str(index) + '_esim06_emodel06.pkl'
with open(filename, 'rb') as out:
	data4 = pickle.load(out)'''
	

		
runs = np.arange(100)

for r in runs:
	data[str(r)]['indices'] = data3[str(r)]['indices']
	for i in range(8):
		data[str(r)]['oracle'].append(data1[str(r)]['oracle'][i])
		data[str(r)]['prob'].append(data1[str(r)]['prob'][i])
		
	for i in range(8):
		data[str(r)]['oracle'].append(data2[str(r)]['oracle'][i])
		data[str(r)]['prob'].append(data2[str(r)]['prob'][i])
		
	for i in range(8):
		data[str(r)]['oracle'].append(data3[str(r)]['oracle'][i])
		data[str(r)]['prob'].append(data3[str(r)]['prob'][i])	
		
	'''for i in range(8):
		data[str(r)]['oracle'].append(data4[str(r)]['oracle'][i])
		data[str(r)]['prob'].append(data4[str(r)]['prob'][i])'''

		
filename = 'cirlfnoisy_' + str(index) + 'bsim0001.pkl'

with open(filename, 'wb') as out:
	pickle.dump(data, out)
		
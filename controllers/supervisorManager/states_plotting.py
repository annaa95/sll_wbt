import numpy as np
import matplotlib.pyplot as plt


name ='data0903202217_24_37.npy'
path ='/home/anna/Documenti/webots_projects/SingleLegLearner/controllers/supervisorManager/data/'
data = np.load(path+name, allow_pickle=True)
t = data[:,0]
ddy = data[:,1]
y = data[:,2]
dx = data[:,3]
fig, axs = plt.subplots(12)
fig.suptitle('Vertically stacked subplots')
for i in range(12):
    axs[i].plot(data[:,i+1])



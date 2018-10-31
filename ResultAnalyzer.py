import pickle
import ToolScripts.Plotter as plotter
import matplotlib.pyplot as plt
import numpy as np
from Params import *

colors = ['red', 'cyan', 'blue', 'green', 'black', 'magenta', 'yellow', 'pink', 'purple', 'chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon', 'gold', 'darkred']
lines = ['-', '--', '-.', ':']





sets = [
	#'59095975973_regWeight_0_BATCH_32_LATENT_DIM300',
	#'29441437673_regWeight_0.001_BATCH_32_LATENT_DIM300',
	#'4213486598_regWeight_0.005_BATCH_32_LATENT_DIM300',
	#'8684693862_regWeight_0.01_BATCH_32_LATENT_DIM300',
	#'10655605038_regWeight_0.005_BATCH_32_LATENT_DIM300',
	#'17324202491_regWeight_0.005_BATCH_256_LATENT_DIM300',
	#'83389457144_regWeight_0.005_BATCH_256_LATENT_DIM300',
	'49266445674_regWeight_0.005_BATCH_256_LATENT_DIM300_LR0.0001_LR_DACAY0.97',
	'28752744673_regWeight_0.005_BATCH_256_LATENT_DIM300_LR0.0005_LR_DACAY0.98'
]
names = [
	#'0',
	#'0.001',#1.925825
	#'0.005',#1.850841
	#'0.01',
	#'0.005-new',
	#'0.005-new-256',
	#'DECAY_0.97',
	'hope',
	'0.0005-0.98'

]



smooth = 1
startLoc = 1
Length = 100
for j in range(len(sets)):
	val = sets[j]
	name = names[j]
	print('val', val)
	# with open('History/%s.his' % val, 'rb') as fs:
	# 	res = pickle.load(fs)
	with open(r'./' + dataset + r'/History/' + val + '.his', 'rb') as fs:
	 	res = pickle.load(fs)

	length = Length
	temy = [None] * 4
	temlength = len(res['loss'])
	temy[0] = np.array(res['loss'][startLoc: min(length, temlength)])
	temy[1] = np.array(res['RMSE'][startLoc: min(length, temlength)])
	temy[2] = np.array(res['val_loss'][startLoc: min(length, temlength)])
	temy[3] = np.array(res['val_RMSE'][startLoc: min(length, temlength)])
	for i in range(4):
		if len(temy[i]) < length-startLoc:
			temy[i] = np.array(list(temy[i]) + [temy[i][-1]] * (length-temlength))
	length -= 1
	y = [[], [], [], []]
	for i in range(int(length/smooth)):
		if i*smooth+smooth-1 >= len(temy[0]):
			break
		for k in range(4):
			temsum = 0.0
			for l in range(smooth):
				temsum += temy[k][i*smooth+l]
			y[k].append(temsum / smooth)
	y = np.array(y)
	length = y.shape[1]
	x = np.zeros((4, length))
	for i in range(4):
		x[i] = np.array(list(range(length)))
	plt.figure(1)
	plt.subplot(221)
	plt.title('LOSS FOR TRAIN')
	plt.plot(x[0], y[0], color=colors[j], label=name)
	plt.legend()
	plt.subplot(222)
	plt.title('LOSS FOR VAL')
	plt.plot(x[2], y[2], color=colors[j], label=name)
	plt.legend()
	plt.subplot(223)
	plt.title('RMSE FOR TRAIN')
	plt.plot(x[1], y[1], color=colors[j], label=name)
	plt.legend()
	plt.subplot(224)
	plt.title('RMSE FOR VAL')
	plt.plot(x[3], y[3], color=colors[j], label=name)
	plt.legend()

plt.show()

import ToolScripts.DataProcessor as proc
import ToolScripts.TimeLogger as logger
import os
import numpy as np
import ToolScripts.Emailer as mailer
import gc
import pickle
from Params import *
import math
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import random
import sys

EpinionsFilterCount = 45818

class ScipyMatMaker:
	def MakeOneMat(self, infile, outfile):
		data = list()
		rows = list()
		cols = list()
		with open(infile, 'r') as fs:
			for line in fs:
				arr = line.strip().split(DIVIDER)
				movieId = int(arr[1]) - 1
				userId = int(arr[0]) - 1
				rating = float(arr[2])
				if MOVIE_BASED:
					rows.append(movieId)
					cols.append(userId)
				else:
					rows.append(userId)
					cols.append(movieId)
				data.append(rating)
		if MOVIE_BASED:
			mat = csr_matrix((data, (rows, cols)), shape=(MOVIE_NUM, USER_NUM))
		else:
			mat = csr_matrix((data, (rows, cols)), shape=(USER_NUM, MOVIE_NUM))
		with open(outfile, 'wb') as fs:
			pickle.dump(mat, fs)

	def MakeTrust(self, infile, outfile):
		data = list()
		rows = list()
		cols = list()
		with open(infile, 'r') as fs:
			for line in fs:
				arr = line.strip().split(DIVIDER)
				tagUserID = int(arr[1]) - 1
				srcUserID = int(arr[0]) - 1
				trustValue = float(arr[2])

				rows.append(srcUserID)
				cols.append(tagUserID)
				data.append(trustValue)

		mat = csr_matrix((data, (rows, cols)), shape=(USER_NUM, USER_NUM))
		with open(outfile, 'wb') as fs:
			pickle.dump(mat, fs)

	def ReadMat(self, file):
		with open(file, 'rb') as fs:
			ret = pickle.load(fs)   
		return ret

	def MakeMats(self):
		trainfile = dataset + '/ratings_' + str(RATE) + '_train.csv'
		testfile = dataset + '/ratings_' + str(RATE) + '_test.csv'
		cvfile = dataset + '/ratings_' + str(RATE) + '_cv.csv'
		trustFile = dataset + '/' + TRUST
		trustOutFile = dataset + '/mats/trust_train.csv' 

		self.MakeOneMat(trainfile, TRAIN_FILE)
		self.MakeOneMat(testfile, TEST_FILE)
		self.MakeOneMat(cvfile, CV_FILE)
		self.MakeTrust(trustFile, trustOutFile)

	

	#过滤掉没有评分的用户
	#return 过滤后的用户数量
	def filter(self):

		with open(TRAIN_FILE, 'rb') as fs:
			train = pickle.load(fs)
		with open(TEST_FILE, 'rb') as fs:
			test = pickle.load(fs)
		with open(CV_FILE, 'rb') as fs:
			cv = pickle.load(fs)
		with open(TRUST_FILE, 'rb') as fs:
			trust = pickle.load(fs)

		allData = train + test + cv
		#找出没有评分的用户索引
		tmp = np.sum(allData!=0, axis=1)
		#tmp2 = np.sum(trust!=0, axis=1)
		#tmp = tmp1 + tmp2
		#找出两个数据集中不全为0的索引  数量45818
		filterIndex = np.where(tmp != 0)[0]
		
		outTrain = train[filterIndex]
		outTest = test[filterIndex]
		outCv = cv[filterIndex]
		outTrust = trust[filterIndex]

		outTmp = outTrust.transpose()
		outTmp = outTmp[filterIndex]
		outTrust = outTmp.transpose()

		with open(TRAIN_FILE, 'wb') as fs:
			pickle.dump(outTrain, fs)
		with open(TEST_FILE, 'wb') as fs:
			pickle.dump(outTest, fs)
		with open(CV_FILE, 'wb') as fs:
			pickle.dump(outCv, fs)
		with open(TRUST_FILE, 'wb') as fs:
			pickle.dump(outTrust, fs)

		return len(filterIndex)


		



class DataDivider:
	def DivideData(self):
		temFile = dataset + '/shuffledRatings.csv'
		proc.RandomShuffle(dataset + '/' + RATING, temFile, True)
		logger.log('End Shuffle')
		rate = RATE
		out1 = dataset + '/ratings_' + str(rate) + '_train.csv'
		out2 = dataset + '/ratings_' + str(rate) + '_test.csv'
		out3 = dataset + '/ratings_' + str(rate) + '_cv.csv'
		proc.SubDataSet(temFile, out1, out2, rate)
		proc.SubDataSet(out1, out1, out3, 0.95 )
		os.remove(temFile)

if __name__ == "__main__":
	logger.log('Start')


	divider = DataDivider()
	divider.DivideData()
	maker = ScipyMatMaker()
	maker.MakeMats()
	filterCount = maker.filter()
	#only Epinions
	if filterCount != EpinionsFilterCount:
		print('warning')
	print(filterCount)
	logger.log('Sparse Matrix Made')
	# mailer.SendMail('No Error')
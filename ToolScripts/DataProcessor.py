import random
import gc

def RandomShuffle(infile, outfile, deleteSchema=False):
	with open(infile, 'r', encoding='utf-8') as fs:
		arr = fs.readlines()
	if not arr[-1].endswith('\n'):
		arr[-1] += '\n'
	if deleteSchema:
		arr = arr[1:]
	random.shuffle(arr)
	with open(outfile, 'w', encoding='utf-8') as fs:
		for line in arr:
			fs.write(line)
	del arr

def SubDataSet(infile, outfile1, outfile2, rate):
	out1 = list()
	out2 = list()
	with open(infile, 'r', encoding='utf-8') as fs:
		for line in fs:
			if random.random() < rate:
				out1.append(line)
			else:
				out2.append(line)
	with open(outfile1, 'w', encoding='utf-8') as fs:
		for line in out1:
			fs.write(line)
	with open(outfile2, 'w', encoding='utf-8') as fs:
		for line in out2:
			fs.write(line)

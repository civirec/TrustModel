# Data Parameters
#dataset = 'ml-20m'
dataset = 'Epinions'
RATE = 0.9

if dataset == 'ml-20m':
	USER_NUM = 138493
	MOVIE_NUM = 26744#150000#26744
	DIVIDER = ','
	RATING = 'ratings.csv'
elif dataset == 'ml-1m':
	USER_NUM = 6040
	MOVIE_NUM = 3706#4000#3706
	DIVIDER = '::'
	RATING = 'ratings.dat'
elif dataset == 'ml-10m':
	USER_NUM = 69878#1000000#69878
	MOVIE_NUM = 10677#500000#10677
	DIVIDER = '::'
	RATING = 'ratings.dat'
elif dataset == 'netflix':
	USER_NUM = 480189#2649429#480189
	MOVIE_NUM = 17770#17770#78305
	DIVIDER = ','
	RATING = 'combined_data_1_new.txt'
elif dataset == 'Epinions':
	USER_NUM = 40163#45818#49290
	MOVIE_NUM = 139738
	DIVIDER = ' '
	RATING = 'ratings_data.txt'
	TRUST = 'trust_data.txt'
MOVIE_BASED = False
DECOMPOSE = False
NECK = 50

# Storage Parameters
#LOAD_MODEL = 'Models/72290089847itemBased_reflect_vreg0.01_wreg0.01_enh0_adam_batch32.model'
#ml-1m original TestRMSE = 0.874381
#LOAD_MODEL = 'Models/87172714180ml-1muserBased_reflect_vreg0.005_wreg0.005_enh0_adam_batch256mats2.model'


TRAIN_FILE = dataset + '/mats/sparseMat_0.9_train.csv'
TEST_FILE = dataset + '/mats/sparseMat_0.9_test.csv'
CV_FILE = dataset + '/mats/sparseMat_0.9_cv.csv'
COL_FILE = dataset + '/mats/sparseMat_0.9_col.csv'
SPARSE_TRAIN = dataset + '/mats/sparsePieces_train'
SPARSE_TEST = dataset + '/mats/sparsePieces_test'
SPARSE_CV = dataset + '/mats/sparsePieces_cv'
DENSE_TRAIN = dataset + '/mats/densePieces_train'
DENSE_TEST = dataset + '/mats/densePieces_test'
DENSE_CV = dataset + '/mats/densePieces_cv'
TRUST_FILE = dataset + '/mats/trust_train.csv'

# Model Parameters
LATENT_DIM = 500
W_WEIGHT = 0.01
V_WEIGHT = 0.01
BATCH_SIZE = 1024
EPOCH = 110
LR = 0.001#
DECAY = 0.96
if MOVIE_BASED:
	DECAY_STEP = MOVIE_NUM / BATCH_SIZE
else:
	DECAY_STEP = USER_NUM / BATCH_SIZE
# DECAY_STEP *= 5
ENHENCE = 0
# testMsg = 'filterColds'

mat = 'mats5'#'mats3'
hashval = 2
for param in dir():
	if not param.startswith('__'):
		val = hash(locals()[param])
		hashval = (hashval * 233 + val) % 100000000007
ModelName = dataset + ('itemBased' if MOVIE_BASED else 'userBased')\
	+ '_reflect_vreg' + str(V_WEIGHT) + '_wreg' + str(W_WEIGHT)\
	+ '_enh' + str(ENHENCE) + '_adam' + '_batch' + str(BATCH_SIZE)\
	+ (('_neck' + str(NECK)) if DECOMPOSE else '')

#print('ModelName', ModelName)

NETFLIX_RATING_LIST = ['','combined_data_1_new.txt', 'combined_data_2_new.txt',
'combined_data_3_new.txt','combined_data_4_new.txt']

MODEL_LIST = ['',
'Models/2568297462netflixuserBased_reflect_vreg0.05_wreg0.05_enh0_adam_batch320mats1.model',
'Models/50109663443netflixuserBased_reflect_vreg0.05_wreg0.05_enh0_adam_batch320mats2.model',
'Models/27872704721netflixuserBased_reflect_vreg0.05_wreg0.05_enh0_adam_batch320mats3.model',
'Models/55991734676netflixuserBased_reflect_vreg0.05_wreg0.05_enh0_adam_batch320mats4.model',
'Models/9558944923netflixuserBased_reflect_vreg0.05_wreg0.05_enh0_adam_batch320mats5.model']




import Params
import datetime


D_in1 = Params.MOVIE_NUM
D_in2 = Params.USER_NUM
K = 300
H = 2 * K
D_out = Params.MOVIE_NUM
LR = 0.0001
regularWeight = 0.005
LR_DECAY = 0.98


BATCH_SIZE = 256

EPOCH = 100


# ISOTIMEFORMAT = r'%Y-%m-%d_%H:%M:%S'
# curTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)

hashval = 2
for param in dir():
	if not param.startswith('__'):
		val = hash(locals()[param])
		hashval = (hashval * 233 + val) % 100000000007


#load model
_ModelName = ''

LOAD_MODEL_PATH = r'./' + Params.dataset + r'/Model/' + _ModelName + r'.pth'


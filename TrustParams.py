import Params
import datetime

D_in1 = Params.MOVIE_NUM
D_in2 = Params.USER_NUM
K = 300
H = 2 * K
D_out = Params.MOVIE_NUM
LR = 0.001
regularWeight = 0.01


BATCH_SIZE = 32

EPOCH = 100


# ISOTIMEFORMAT = r'%Y-%m-%d_%H:%M:%S'
# curTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)

hashval = 2
for param in dir():
	if not param.startswith('__'):
		val = hash(locals()[param])
		hashval = (hashval * 233 + val) % 100000000007


#load model
_ModelName = '29441437673_regWeight_0.001_BATCH_32_LATENT_DIM300'

LOAD_MODEL_PATH = r'./' + Params.dataset + r'/Model/' + _ModelName + r'.pth'


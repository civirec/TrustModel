import torch as t
import torch.nn as nn
import torch.nn.functional as F
from TrustParams import *
#import torch.optim as optim
#import Params
#from MakeTrustData import ScipyMatMaker

device = t.device("cuda")


class TrustNet(nn.Module):
    def __init__(self):#, inDim1, inDim2, hideDim, outDim, batch):
        super(TrustNet, self).__init__()

        self.inputRating = nn.Linear(D_in1, K)
        #self.inputTrust = nn.Linear(D_in2, K)
        #TODO 
        self.inputTrust = nn.Linear(40163, K)
        self.out = nn.Linear(2 * K, D_out)

        # self.model = nn.Sequential(
        #     nn.Linear(inDim, hideDim, bias=False),
        #     nn.Sigmoid(),
        #     nn.Linear(hideDim, outDim, bias=False),
        # )
        
    def forward(self, rating, trust):
        x1 = t.sigmoid(self.inputRating(rating))
        x2 = t.sigmoid(self.inputTrust(trust))
        x = t.cat((x1, x2), dim=1) #列方向连接
        y_pred = self.out(x)
        return y_pred
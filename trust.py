import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import Params
from MakeTrustData import ScipyMatMaker
from TrustParams import *
from TrustNet import TrustNet
from ToolScripts.TimeLogger import log
import pickle
import torch.utils.data



device = t.device("cuda")

'''
x1 = t.randn(BATCH_SIZE, D_in1, device = device)
(x1)
x2 = t.randn(BATCH_SIZE, D_in2, device = device)
print(x2)
y = t.randn(BATCH_SIZE, D_out,device = device)
print(y)
class TrustNet(nn.Module):
    def __init__(self, inDim1, inDim2, hideDim, outDim, batch):
        super(TrustNet, self).__init__()
        # self.inDim = inDim
        # self.hideDim = hideDim
        # self.outDim = outDim
        # self.batch = batch

        self.inputRating = nn.Linear(D_in1, K)
        self.inputTrust = nn.Linear(D_in2, K)
        self.out = nn.Linear(2 * K, D_out)

        # self.model = nn.Sequential(
        #     nn.Linear(inDim, hideDim, bias=False),
        #     nn.Sigmoid(),
        #     nn.Linear(hideDim, outDim, bias=False),
        # )
        
    def forward(self, rating, trust):
        x1 = F.sigmoid(self.inputRating(rating))
        x2 = F.sigmoid(self.inputTrust(trust))
        x = t.cat((x1, x2), dim=1) #列方向连接
        y_pred = self.out(x)
        return y_pred
'''

class MyModel():

    def __init__(self, train, test, cv, trust, lr, regWeight, isLoadModel=False):
        #传入数据
        self.trainMat = train
        self.testMat = test
        self.cvMat = cv
        self.trustMat = trust
        self.trusteeMat = trust.transpose()

        self.trainMask = trainMat != 0
        self.cvMask = cvMat != 0
        self.testMask = testMat != 0
        #创建模型
        self.model = TrustNet().cuda()
        self.loss_rmse = nn.MSELoss(reduction='sum')#不求平均
        #self.loss = nn.MSELoss()
        self.learning_rate = lr #0.001
        self.regWeight = regWeight
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regWeight)
        #self.optimizer = t.optim.SGD(self.model.parameters(), lr=self.learning_rate,weight_decay=self.regWeight)
        self.curEpoch = 0
        self.isLoadModel = isLoadModel
        #历史数据，用于画图
        self.train_losses = []
        self.train_RMSEs  = []
        self.test_losses  = []
        self.test_RMSEs   = []

    def run(self):
        #判断是导入模型还是重新训练模型
        if self.isLoadModel == True:
            self.loadModel(LOAD_MODEL_PATH)
        for e in range(self.curEpoch, EPOCH+1):
            #训练
            epoch_loss, epoch_rmse = self.trainModel(self.trainMat, self.trustMat, self.trusteeMat, self.trainMask, self.optimizer)
            regularLoss = self.getRegularLoss(self.model)
            log("\n")
            log("epoch %d/%d, epoch_loss=%.2f, epoch_rmse=%.4f"%(e,EPOCH, epoch_loss, epoch_rmse))
            self.train_losses.append(epoch_loss)
            self.train_RMSEs.append(epoch_rmse)
            #打印正则损失
            log('W1_Loss = %.2f, W2_Loss = %.2f, W3_Loss = %.2f'%(regularLoss[0], regularLoss[1], regularLoss[2]))

            #交叉验证
            cv_epoch_loss, cv_epoch_rmse = self.testModel(self.trainMat, self.trustMat, self.trusteeMat, self.cvMat, self.cvMask, 5)
            log("\n")
            log("epoch %d/%d, cv_epoch_loss=%.2f, cv_epoch_rmse=%f"%(e, EPOCH, cv_epoch_loss, cv_epoch_rmse))
            self.test_losses.append(cv_epoch_loss)
            self.test_RMSEs.append(cv_epoch_rmse)

            #每一个Epoch调整学习率
            self.adjust_learning_rate(self.optimizer, e)
            #记录当前epoch,用于保存Model
            self.curEpoch = e
            if e % 5 == 0 and e != 0:
                self.saveModel()
               
        #测试
        test_epoch_loss, test_epoch_rmse = self.testModel(self.trainMat, self.trustMat, self.trusteeMat, self.testMat, self.testMask)
        log("\n")
        log("test_rmse=%f"%(test_epoch_rmse))

    #user-regular,bias,trust-regular,bias,combine-regular,bias
    def getRegularLoss(self, model):
        regularLoss = []
        tmp = []
        with torch.no_grad():
            for param in self.model.parameters():
                data = param.data
                loss = t.sum(data * data).item()
                tmp.append(loss)
        regularLoss.append(tmp[0] + tmp[1])
        regularLoss.append(tmp[2] + tmp[3])
        regularLoss.append(tmp[4] + tmp[5])
        return regularLoss

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.learning_rate * LR_DECAY
        #lr = self.learning_rate * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def getModelName(self):
        ModelName = str(hashval) + \
        "_regWeight_" + str(self.regWeight) + \
        "_BATCH_" + str(BATCH_SIZE) + \
        "_LATENT_DIM" + str(K) + \
        "_LR" + str(LR) + \
        "_LR_DACAY" + str(LR_DECAY)
        return ModelName



    def trainModel(self, trainMat, trustMat, trusteeMat, trainMask, op):
        num = trainMat.shape[0]
        shuffledIds = np.random.permutation(num)
        steps = int(np.ceil(num / BATCH_SIZE))
        epoch_loss = 0
        epoch_rmse = 0
        epoch_rmse_num = 0
        for i in range(steps):
            ed = min((i+1) * BATCH_SIZE, num)
            batch_ids = shuffledIds[i * BATCH_SIZE: ed]
            cur_batch_size = len(batch_ids)
            tmpTrain  = trainMat[batch_ids].toarray()
            tmpTrust  = trustMat[batch_ids].toarray()
            tmpTrustee = trusteeMat[batch_ids].toarray()
            #fakeTrust = np.zeros_like(tmpTrust)
            tmpMask   = trainMask[batch_ids].toarray()

            train = t.from_numpy(tmpTrain).float().to(device)
            trust = t.from_numpy(tmpTrust).float().to(device)
            trustee = t.from_numpy(tmpTrustee).float().to(device)
            mask = t.from_numpy(1*tmpMask).float().to(device) #将bool转为int,否则会报错

            y_pred = self.model(train, trust, trustee)
            #loss = self.loss(y_pred * mask, train)

            loss = self.loss_rmse(y_pred*mask, train)/cur_batch_size

            epoch_loss += loss.item()

            #tem = (y_pred * mask - train)
            
            #a = t.sum(tem * tem).item()
            #tem = self.loss_rmse(y_pred * mask, train)
            tem = loss * cur_batch_size

            epoch_rmse += tem.item()
            epoch_rmse_num += t.sum(mask).item()
            
            log('setp %d/%d, step_loss = %f'%(i,steps, loss), save=False, oneline=True)

            op.zero_grad()
            loss.backward()

            
            op.step()
        epoch_rmse = np.sqrt(epoch_rmse / epoch_rmse_num)
        #epoch_loss = epoch_loss / steps
        return epoch_loss, epoch_rmse

    def testModel(self, trainMat, trustMat, trusteeMat, testLabel, testMask, div=1):
        shuffledIds = np.random.permutation(trainMat.shape[0])
        steps = int(np.ceil(trainMat.shape[0] /(BATCH_SIZE * div)))
        epoch_loss = 0
        epoch_rmse = 0
        epoch_rmse_num = 0
        for i in range(steps):
            ed = min((i+1) * BATCH_SIZE, trainMat.shape[0])
            batch_ids = shuffledIds[i * BATCH_SIZE: ed]
            cur_batch_size = len(batch_ids)
            tmpTrain = trainMat[batch_ids].toarray()
            tmpTrust = trustMat[batch_ids].toarray()
            tmpTrustee = trusteeMat[batch_ids].toarray()
            #fakeTrust = np.zeros_like(tmpTrust)
            tmpLabel = testLabel[batch_ids].toarray()
            tmpMask = testMask[batch_ids].toarray()
            
            train = t.from_numpy(tmpTrain).float().to(device)
            trust = t.from_numpy(tmpTrust).float().to(device)
            trustee = t.from_numpy(tmpTrustee).float().to(device)
            label = t.from_numpy(tmpLabel).float().to(device)
            mask = t.from_numpy(1*tmpMask).float().to(device) #将bool转为int,否则会报错

            y_pred = self.model(train, trust, trustee)

            loss = self.loss_rmse(y_pred * mask, label)/cur_batch_size

            tem = loss * cur_batch_size
            #tem = (y_pred * mask - label)
            #epoch_rmse += t.sum(tem * tem).item()
            epoch_loss += loss.item()
            epoch_rmse += tem.item()
            epoch_rmse_num += t.sum(mask).item()
        epoch_rmse = np.sqrt(epoch_rmse / epoch_rmse_num)
        return epoch_loss, epoch_rmse


    def saveModel(self):
        ModelName = self.getModelName()
        savePath = r'./' + Params.dataset + r'/Model/' + ModelName + r'.pth'
        #t.save(self.model, savePath)
        t.save({
            'epoch': self.curEpoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, savePath)
        print("save model : " + ModelName)
        #保存历史数据，用于画图
        history = dict()
        history['loss'] = self.train_losses
        history['RMSE'] = self.train_RMSEs
        history['val_loss'] = self.test_losses
        history['val_RMSE'] = self.test_RMSEs    

        with open(r'./' + Params.dataset + r'/History/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    

    def loadModel(self, modelPath):
        self.model = TrustNet().cuda()
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regWeight)
        checkpoint = t.load(modelPath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curEpoch = checkpoint['epoch']
        self.model.eval()

    def rmse(self, pred, label):
        return t.sqrt(self.loss_fn(pred, label))

    def showModel(self):
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
    	    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

	    # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
    	    print(var_name, "\t", self.optimizer.state_dict()[var_name])



    #获取数据
def prepareData():
    maker = ScipyMatMaker()
    trainMat = maker.ReadMat(Params.TRAIN_FILE)
    testMat = maker.ReadMat(Params.TEST_FILE)
    cvMat = maker.ReadMat(Params.CV_FILE)
    trustMat = maker.ReadMat(Params.TRUST_FILE)
    #trustMat = maker.ReadMat('Epinions/mats/trust_train.csv')
    return trainMat, testMat, cvMat, trustMat



if __name__ == '__main__':

    trainMat, testMat, cvMat, trustMat = prepareData()

    hope = MyModel(trainMat, testMat, cvMat, trustMat, LR, regularWeight, isLoadModel=False)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    loss_fn = nn.MSELoss()

    hope.run()
'''
    # L2 norm 权重
    # optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=0.05)

	

    # print(model)
    # print(model.parameters())
    # for param in model.parameters():
    #     print(type(param.data), param.size())
    #     print(param)

    #t.save(model.state_dict(), './Models/1.model')
    # for e in range(EPOCH):
    #     y_pred = model(x1, x2)

    #     #rmse
    #     loss = t.sqrt(loss_fn(y_pred, y))
    #     print(e, loss.item())

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()



    # model = nn.Sequential(
    #     nn.Linear(D_in, H),
    #     nn.Sigmoid(),
    #     nn.Linear(H, D_out),
    # )
    # model = model.cuda()

    # loss_fn = nn.MSELoss(reduction='sum')

    # learning_rate = 1e-4
    # optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    # for epoch in range(1000):
        
    #     y_pred = model(x)

    #     loss = loss_fn(y_pred, y)
    #     print(epoch, loss.item())

    #     optimizer.zero_grad()

    #     loss.backward()

    #     optimizer.step()
'''



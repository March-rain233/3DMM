from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import glob
import time
from math import cos, sin
import scipy.io as sio
from PIL import Image
import numpy as np
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
from lib.Model import FitModel

# 超参数
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备
BATCH_SIZE = 32  # 批大小
EPOCH_NUM = 40001  # 训练次数 
LR = 5e-5  # 学习率
CLIP = 5  # 梯度裁剪
SPLIT = 0.75  # 训练集与测试集分割率
DATASET_PATH = r'data\300W_LP\LFPW/'  # 训练集根目录
EXTENSION = 'jpg'  # 图片后缀名
LOG_PATH = None  # 日志路径
SAVE_PATH = r'model\fit_model'  # 模型保存路径
MODEL_NAME = 'fit_model'  # 模型名
LAST_EPOCH = 690  # 上一次运行的EPOCH
PER_EPOCH_SAVE = 10  # 每多少EPOCH储存一次
PER_EPOCH_TEST = 2  # 每多少EPOCH测试一次
SEED = 163  # 分隔训练集随机种子
INPUT_SIZE = 120  # 输入图像大小

BFM_PATH = r'model/BFM.mat'  # BFM模型位置
SHAPE_NUM = 40  # 形状参数数量
EXP_NUM = 10  # 表情参数数量
PR = torch.as_tensor([[1, 0, 0],
                      [0, 1, 0]], dtype=torch.float32, device=DEVICE)  # 正交投影矩阵
L2DCON_IND =        torch.as_tensor([18,20,22,23,25,27,37,41,40,43,48,46,32,34,36,49,63,55]) - 1
L2DCON_WEIGHTS =    torch.as_tensor([1 ,1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 4, 1, 2, 4, 2], dtype=torch.float32, device=DEVICE)
L2DCON_WEIGHTS /= L2DCON_WEIGHTS.sum()

L3D_WEIGHTS = torch.ones([12 + SHAPE_NUM + EXP_NUM], dtype=torch.float32, device=DEVICE)
L3D_WEIGHTS[0] = 1
L3D_WEIGHTS[1:3] = 1
L3D_WEIGHTS[3:12] = 1
L3D_WEIGHTS /= L3D_WEIGHTS.sum()

LAMBDAS = [0.005, 0.005, 1, 0.005]  # 损失函数权重

class CustomLoss(nn.Module):
    def __init__(self, lambda1=1, lambda2=3):
        super(CustomLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, x, y):
        ys = torch.sign(y)
        yp = y * ys
        xp = x * ys
        ymax = torch.max(yp, xp)
        return self.lambda1 * F.mse_loss(yp, ymax) + self.lambda2 * F.mse_loss(xp, ymax)

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def projection(vertexes, f, pi, t):
    '''
    Args:
        vertexes: [batcn_size, nver, 3]
        f: [batch_size,]
        pi: [batch_size, 9]
        t: [batch_size, 2]
    Returns:
        x2d: [batch_size, nver, 2]
    '''
    pi = pi.reshape(-1, 3, 3)  # [batch_size, 3, 3]
    vertexes = vertexes.permute(0, 2, 1)  # [batch_size, 3, nver]
    f = f[:, None, None]  # [batch_size, 1, 1]
    t = t[..., None]  # [batch_size, 2, 1]
    out = pi @ vertexes  # [batch_size, 3, nver]
    out = PR @ out  # [batch_size, 2, nver]
    out = f * out  # [batch_size, 2, nver]
    out = out + t  # [batch_size, 2, nver]
    return out.permute(0, 2, 1)  # [batch_size, nver, 2]

class ImageDataset(Dataset):
    def __init__(self, filePath=r'data\train_data\*', extension='jpg') -> None:
        super().__init__()
        self.imagePaths = glob.glob(filePath + "." + extension)
        self.labelPaths = glob.glob(filePath + ".mat")
        self.transforms = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    
    def __getitem__(self, index):
        x = Image.open(self.imagePaths[index]).convert('RGB')
        x = self.transforms(x)
        label = sio.loadmat(self.labelPaths[index])
        s = label['Pose_Para'][0, -1]
        angles = label['Pose_Para'][0, :3]
        t = label['Pose_Para'][0, 3:6]
        pi = angle2matrix(angles)
        y = torch.as_tensor(np.r_[
            s,
            t[:2],
            pi.flatten(),
            label['Shape_Para'][:SHAPE_NUM, 0],
            label['Exp_Para'][:EXP_NUM, 0],
        ].flatten(), dtype=torch.float32)

        x2d = (label['pt2d'] * INPUT_SIZE / 450).astype(np.int32)
        flms = torch.full([1, x.shape[1], x.shape[2]], -1)
        flms[0, x2d[1], x2d[0]] = 1
        return x, flms, y
    
    def __len__(self):
        return len(self.imagePaths)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Bilinear') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias != None: nn.init.zeros_(tensor=m.bias)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
        if m.bias != None: nn.init.zeros_(tensor=m.bias)

    elif classname.find('BatchNorm') != -1 or classname.find('GroupNorm') != -1 or classname.find('LayerNorm') != -1:
        nn.init.uniform_(a=0, b=1, tensor=m.weight)
        nn.init.zeros_(tensor=m.bias)

    elif classname.find('Cell') != -1:
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_hh)
        nn.init.xavier_uniform_(gain=1, tensor=m.weight_ih)
        nn.init.ones_(tensor=m.bias_hh)
        nn.init.ones_(tensor=m.bias_ih)

    elif classname.find('RNN') != -1 or classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for w in m.all_weights:
            nn.init.xavier_uniform_(gain=1, tensor=w[2].data)
            nn.init.xavier_uniform_(gain=1, tensor=w[3].data)
            nn.init.ones_(tensor=w[0].data)
            nn.init.ones_(tensor=w[1].data)

    if classname.find('Embedding') != -1:
        nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)

def getMeanAndVar(dataLoder):
    mean = 0
    var = 0
    l = len(dataLoder)
    for x, flms, y in dataLoder:
        var_t, mean_t = torch.var_mean(y)
        mean += mean_t.item()
        var += var_t.item()
    mean /= l
    var /= l
    return mean, var

def l2dcon(x, y):
    '''
    Args:
        x: [batch_size, nver, 2]
        y: [batch_size, nver, 2]
    '''
    x = x[:,L2DCON_IND,:]
    y = y[:,L2DCON_IND,:]
    loss = F.mse_loss(x, y, reduction='none')
    loss = loss * L2DCON_WEIGHTS[None, :, None]
    return loss.mean()

def main():
    # 数据集载入
    dataset =ImageDataset(DATASET_PATH + '*train*', EXTENSION)
    print(f'成功从{DATASET_PATH}载入数据集，数据集大小为: {len(dataset)}')
    generator = torch.Generator().manual_seed(SEED)
    trainSet, testSet = dataset, ImageDataset(DATASET_PATH + '*test*', EXTENSION) #random_split(dataset, [SPLIT, 1 - SPLIT], generator)
    print(f'以{SPLIT}比率拆分数据集，训练集大小: {len(trainSet)} 测试集大小: {len(testSet)}')
    trainLoader = DataLoader(trainSet, BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(testSet, BATCH_SIZE)

    trainMean, trainVar = getMeanAndVar(trainLoader)
    testMean, testVar = getMeanAndVar(testLoader)

    #BFM模型载入
    bfmdata = sio.loadmat(BFM_PATH)['model']
    kptInd = bfmdata['kpt_ind'][0, 0][0] - 1 
    indices = kptInd.astype(np.uint32) * 3
    indices = np.c_[indices, indices + 1, indices + 2].flatten()
    shapeMu = torch.as_tensor(bfmdata['shapeMU'][0, 0].flatten()[indices],      dtype=torch.float32, device=DEVICE)
    shapePC = torch.as_tensor(bfmdata['shapePC'][0, 0][:, :SHAPE_NUM][indices], dtype=torch.float32, device=DEVICE)
    expMU = torch.as_tensor(bfmdata['expMU'][0, 0].flatten()[indices],          dtype=torch.float32, device=DEVICE)
    expPC = torch.as_tensor(bfmdata['expPC'][0, 0][:, :EXP_NUM][indices],       dtype=torch.float32, device=DEVICE)
    
    getVertexes = lambda sp, ep :(shapeMu + (shapePC @ sp.T + expPC @ ep.T).T).reshape(sp.shape[0], -1, 3)

    # 训练前准备
    net = FitModel(SHAPE_NUM, EXP_NUM)
    if LAST_EPOCH != -1:
        net.load_state_dict(torch.load(SAVE_PATH + rf'\{MODEL_NAME}-{LAST_EPOCH}.pth'))
    else:
        net.apply(init_weights)

    optimizer = optim.Adam([{'params':net.parameters(), 'initial_lr':LR}], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, last_epoch=LAST_EPOCH)

    with SummaryWriter(LOG_PATH, comment=f'-Lr_{LR:e}-Bacth_{BATCH_SIZE}', purge_step=LAST_EPOCH) as writer:
        net.to(DEVICE)

        # 训练模型
        testing_r2 = 0
        print(f'开始训练\nMaxEpoch: {EPOCH_NUM} BatchSize: {BATCH_SIZE} LearnRate: {LR} Device: {DEVICE}')
        for epoch in range(LAST_EPOCH + 1, EPOCH_NUM):
            start = time.time()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('train/lr', lr, epoch)
            running_loss = 0.0
            running_mse = 0.0

            net.train()
            for i, data in enumerate(trainLoader, 0):
                inputs, flms, labels = data
                inputs, flms, labels = inputs.to(DEVICE), flms.to(DEVICE), labels.to(DEVICE)

                x1 = torch.cat([inputs, flms], 1)
                optimizer.zero_grad()
                outputs = net(x1)

                x2d = projection(getVertexes(labels[:,12:12+SHAPE_NUM], labels[:,12+SHAPE_NUM:])
                                 , labels[:,0], labels[:,3:12], labels[:,1:3])  # [batch_size, nver, 2]
                x3d = getVertexes(outputs[:,12:12+SHAPE_NUM], outputs[:,12+SHAPE_NUM:])  # [batch_size, nver, 3]
                y2d = projection(x3d, outputs[:,0], outputs[:,3:12], outputs[:,1:3])  # [batch_size, nver, 2]

                flms2 = torch.full_like(flms, -1)
                ind = torch.clamp(y2d.int(), 0, INPUT_SIZE - 1)
                indx = ind[:,:,0].flatten()
                indy = ind[:,:,1].flatten()
                indn = torch.arange(y2d.shape[0])[:,None].repeat(1, y2d.shape[1]).flatten()
                flms2[indn, :, indy, indx] = 1

                x2 = torch.cat([inputs, flms2], 1)
                outputs2 = net(x2)

                x3dHat = getVertexes(outputs2[:,12:12+SHAPE_NUM], outputs2[:,12+SHAPE_NUM:])
                x2dHat = projection(x3dHat, outputs2[:,0], outputs2[:,3:12], outputs2[:,1:3])


                loss3d = F.mse_loss(outputs, labels, reduction='none')
                loss3d = loss3d * L3D_WEIGHTS[None, :]
                loss3d = loss3d.mean()
                loss2dcon = l2dcon(x2d, y2d)
                loss3dcon = F.mse_loss(x3d, x3dHat)
                losscyc = l2dcon(x2d, x2dHat)
                loss = loss3d + LAMBDAS[0] * loss2dcon + LAMBDAS[1] * loss3dcon + LAMBDAS[2] * losscyc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
                optimizer.step()

                running_mse += F.mse_loss(outputs, labels).item()
                running_loss += loss.item()

            scheduler.step()

            running_loss /= len(trainLoader)
            running_mse /= len(trainLoader)
            running_r2 = 1 - running_mse / trainVar
            writer.add_scalar('train/loss', running_loss, epoch)
            writer.add_scalar('train/R2/train', running_r2, epoch)
            writer.add_images('x2d&y2d', torch.cat([flms, flms2], dim=3), epoch)

            if epoch % PER_EPOCH_TEST == 0:
                net.eval()
                testing_mse = 0
                with torch.no_grad():
                    for X, flms, Y in testLoader:
                        X, flms, Y = X.to(DEVICE), flms.to(DEVICE), Y.to(DEVICE)
                        X = torch.cat([X, flms], 1)
                        pred = net(X)
                        testing_mse += F.mse_loss(pred, Y).item()

                testing_mse /= len(testLoader)
                testing_r2 = 1 - testing_mse / testVar
                writer.add_scalar('train/R2/test', testing_r2, epoch)

            end = time.time()
            print(f'Epoch: {epoch:<6}  用时: {end - start:<.2f}s  Loss: {running_loss:<6e}  Train-R2: {running_r2:.6f}  Test-R2: {testing_r2:.6f}  Lr: {lr:<6e}')

            # 保存暂时文件
            if epoch != 0 and epoch % PER_EPOCH_SAVE == 0:
                torch.save(net.state_dict(), SAVE_PATH + rf'\{MODEL_NAME}-{epoch}.pth')

        print('训练完成')

        # 保存模型
        torch.save(net.state_dict(), SAVE_PATH + rf'\{MODEL_NAME}.pth')


if __name__=='__main__':
    main()
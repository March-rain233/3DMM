import h5py
import numpy as np
import json

class MorphabelModel:
    def __init__(self, filePath, maxIter=100):
        with h5py.File(filePath, 'r') as f:
            self.shapeMU = f['shape/model/mean'][:]   #平均人脸形状
            self.shapePC = f['shape/model/pcaBasis'][:]   #形状主成分
            self.shapeEV = f['shape/model/pcaVariance'][:]  #形状主成分方差
            self.tl = f['shape/representer/cells'][:].T   #三角面片
            self.texMU = f['color/model/mean'][:]   #平均人脸纹理
            self.texPC = f['color/model/pcaBasis'][:]   #纹理主成分
            self.texEV = f['color/model/pcaVariance'][:]  #纹理主成分方差
            self.expMU = f['expression/model/mean'][:]  #平均人脸表情
            self.expPC = f['expression/model/pcaBasis'][:]  #平均人脸表情
            self.expEV = f['expression/model/pcaVariance'][:]   #表情主成分方差
            self.kptInd = json.loads(f['metadata/landmarks/json'][0])   #特征点

            self.nVer = self.shapeMU.shape[0] / 3   #顶点数量
            self.nTri = self.tl.shape[0]  #三角面片数量
            self.nSP = self.shapePC.shape[1]  #形状主成分变量数
            self.nTP = self.texPC.shape[1]  #纹理主成分变量数
            self.nEP = self.expPC.shape[1]  #表情主成分变量数

            self.sp = np.zeros(self.nSP)  #形状主成分变量
            self.tp = np.zeros(self.nTP)  #纹理主成分变量
            self.ep = np.zeros(self.nEP)  #表情主成分变量
        self.maxIter = maxIter  #最大迭代次数

    def transform(self):
        return ((self.shapeMU 
                  + self.shapePC @ self.sp 
                  + self.expPC @ self.ep
                  ).reshape(-1,3).astype(np.float32),
                 (self.texMU 
                  + self.texPC @ self.tp
                  ).reshape(-1, 3).astype(np.float32))
    
    def fit(self):
        pass
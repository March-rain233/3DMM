import h5py
import numpy as np
import json
import scipy.io as sio

class MorphabelModel:
    def __init__(self, filePath, maxIter=100):
        data = sio.loadmat(filePath)['model']
        self.shapeMU = data['shapeMU'][0, 0].T                  #平均人脸形状
        self.shapePC = data['shapePC'][0, 0]                    #形状主成分
        self.shapeEV = data['shapeEV'][0, 0]                    #形状主成分方差
        self.expMU = data['expMU'][0, 0].T                      #平均人脸表情
        self.expPC = data['expPC'][0, 0]                        #表情主成分
        self.expEV = data['expEV'][0, 0]                        #表情主成分方差
        self.texMU = data['texMU'][0, 0].T                      #平均人脸纹理
        self.texPC = data['texPC'][0, 0]                        #纹理主成分
        self.texEV = data['texEV'][0, 0]                        #纹理主成分方差
        self.tri = (data['tri'][0, 0].T - 1).astype(np.uint32)  #三角面坐标
        self.kptInd = data['kpt_ind'][0, 0][0]                  #特征点
        self.nVer = self.shapeMU.shape[1] / 3                   #顶点数量
        self.nTri = self.tri.shape[0]                           #三角面片数量
        self.nSP = self.shapePC.shape[1]                        #形状主成分变量数
        self.nTP = self.texPC.shape[1]                          #纹理主成分变量数
        self.nEP = self.expPC.shape[1]                          #表情主成分变量数
        self.sp = np.zeros(self.nSP)                            #形状主成分变量
        self.tp = np.zeros(self.nTP)                            #纹理主成分变量
        self.ep = np.zeros(self.nEP)                            #表情主成分变量
        self.maxIter = maxIter                                  #最大迭代次数

    def transform(self):
        return ((self.shapeMU 
                  + self.shapePC @ self.sp 
                  + self.expPC @ self.ep
                  ).reshape(-1,3).astype(np.float32),
                 ((self.texMU 
                  + self.texPC @ self.tp
                  ) / 255).reshape(-1, 3).astype(np.float32))
    
    def getPAffine(self, x2d, x3d):
        '''获取仿射矩阵'''

        n = x2d.shape[0]
        #normalization
        #处理2维点
        #平移所有坐标点，使它们的质心位于原点
        mean = x2d.mean(0)
        x2d = x2d - mean
        #对这些点进行缩放，使到原点的平均距离等于根号二
        avgDIs = np.mean(np.linalg.norm(x2d, axis=1))
        scale = np.sqrt(2) / avgDIs
        x2d = x2d * scale
        #获取变化矩阵T
        T = np.identity(3, dtype = np.float32)
        T[0, 0] = T[1, 1] = scale
        T[:2, 2] = -mean * scale

        #处理3维点
        #平移所有坐标点，使它们的质心位于原点
        mean = x3d.mean(0)
        x3d = x3d - mean
        #对这些点进行缩放，使到原点的平均距离等于根号三
        avgDIs = np.mean(np.linalg.norm(x3d, axis=1))
        scale = np.sqrt(3) / avgDIs
        x3d = x3d * scale
        #获取变化矩阵U
        U = np.identity(4, dtype = np.float32)
        U[0, 0] = U[1, 1] = U[2, 2] = scale
        U[:3, 3] = -mean * scale

        #equations
        A = np.zeros((n * 2, 8), dtype=np.float32)
        xhat = np.c_[x3d, np.zeros(n)]
        A[:n, :4] = xhat
        A[n:, 4:] = xhat
        b = x2d.flatten('f')

        #solution
        p8 = np.linalg.pinv(A) @ b
        P = np.zeros([3, 4])
        P[0, :] = p8[:4]
        P[1, :] = p8[4:]
        P[2, 3] = 1

        #denormalization
        return np.linalg.pinv(T) @ P @ U

    def fit(self, kptPoints):
        pass
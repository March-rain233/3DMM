import h5py
import numpy as np
import json
import scipy.io as sio

class MorphabelModel:
    def __init__(self, filePath, maxIter=5):
        data = sio.loadmat(filePath)['model']
        self.shapeMU = data['shapeMU'][0, 0].flatten()          #平均人脸形状
        self.shapePC = data['shapePC'][0, 0]                    #形状主成分
        self.shapeEV = data['shapeEV'][0, 0]                    #形状主成分方差
        self.expMU = data['expMU'][0, 0].flatten()              #平均人脸表情
        self.expPC = data['expPC'][0, 0]                        #表情主成分
        self.expEV = data['expEV'][0, 0]                        #表情主成分方差
        self.texMU = data['texMU'][0, 0].flatten()              #平均人脸纹理
        self.texPC = data['texPC'][0, 0]                        #纹理主成分
        self.texEV = data['texEV'][0, 0]                        #纹理主成分方差
        self.tri = (data['tri'][0, 0].T - 1).astype(np.uint32)  #三角面坐标
        self.kptInd = data['kpt_ind'][0, 0][0] - 1              #特征点
        self.nVer = self.shapeMU.shape[0] / 3                   #顶点数量
        self.nTri = self.tri.shape[0]                           #三角面片数量
        self.nSP = self.shapePC.shape[1]                        #形状主成分变量数
        self.nTP = self.texPC.shape[1]                          #纹理主成分变量数
        self.nEP = self.expPC.shape[1]                          #表情主成分变量数
        self.sp = np.random.rand(self.nSP)*1e04                 #形状主成分变量
        self.tp = np.random.rand(self.nTP)                      #纹理主成分变量
        self.ep = -1.5 + 3*np.random.random(self.nEP)           #表情主成分变量
        self.maxIter = maxIter                                  #最大迭代次数
        self.shapeMU = self.shapeMU + self.expMU

    def transform(self):
        return ((self.shapeMU 
                  + self.shapePC @ self.sp 
                  + self.expPC @ self.ep
                  ).reshape(-1,3).astype(np.float32),
                 ((self.texMU 
                  + self.texPC @ self.tp
                  ) / 255).reshape(-1, 3).astype(np.float32))
    
    def fit(self, kptPoints):
        '''校准'''
        #截取特征点对应的参数
        indices = self.kptInd.astype(np.uint32) * 3
        indices = np.c_[indices, indices + 1, indices + 2].flatten()
        shapeMU = self.shapeMU[indices]
        shapePC = self.shapePC[indices]
        expPC = self.expPC[indices]


        #迭代校准
        for i in range(self.maxIter):
            sp = self.sp
            ep = self.ep

            shape = shapePC @ self.sp

            x3d = (shapeMU + shape + expPC @ self.ep).reshape(-1, 3)
            PA = self.__getPAffine(kptPoints, x3d)
            s, R, t = self.__P2SRT(PA)
            self.ep = self.__estimateExpression(kptPoints.T, shapeMU, expPC, self.expEV, shape.reshape(-1, 3).T, s, R, t[:2], 
                                                lamb = 100).flatten()

            exp = expPC @ self.ep
            self.sp = self.__estimateShape(kptPoints.T, shapeMU, shapePC, self.shapeEV, exp.reshape(-1, 3).T, s, R, t[:2], 
                                                lamb = 1).flatten()
        #print(f'sp最大更改:{np.max(np.abs(self.sp - sp))}')
        #print(f'np最大更改:{np.max(np.abs(self.ep - ep))}')
    
    def __getPAffine(self, x2d, x3d):
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

    def __P2SRT(self, P):
        '''仿射矩阵求S、R、T'''
        t = P[:, 3]
        r1 = P[0, :3]
        r2 = P[1, :3]
        nr1 = np.linalg.norm(r1)
        nr2 = np.linalg.norm(r2)
        s = (nr1 + nr2) / 2
        r1 = r1 / nr1
        r2 = r2 / nr2
        r3 = np.cross(r1, r2)
        R = np.c_[(r1, r2, r3)].T
        return s, R, t

    def __estimateExpression(self, x, shapeMU, expPC, expEV, shape, s, R, t2d, lamb):
        '''
        Args:
            x: (2, n). image points (to be fitted)
            shapeMU: (3n, 1)
            expPC: (3n, n_ep)
            expEV: (n_ep, 1)
            shape: (3, n)
            s: scale
            R: (3, 3). rotation matrix
            t2d: (2,). 2d translation
            lambda: regulation coefficient

        Returns:
            exp_para: (n_ep, 1) shape parameters(coefficients)
        '''
        x = x.copy()
        assert(shapeMU.shape[0] == expPC.shape[0])
        assert(shapeMU.shape[0] == x.shape[1]*3)

        dof = expPC.shape[1]

        n = x.shape[1]
        sigma = expEV
        t2d = np.array(t2d)
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        A = s*P.dot(R) #(2,3)

        # --- calc pc
        pc_3d = np.resize(expPC.T, [dof, n, 3]) 
        pc_3d = np.reshape(pc_3d, [dof*n, 3]) # (29n,3)
        pc_2d = pc_3d.dot(A.T) #(29n,2)
        pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

        # --- calc b
        # shapeMU
        mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
        # expression
        shape_3d = shape
        # 
        b = A.dot(mu_3d + shape_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
        b = np.reshape(b.T, [-1, 1]) # 2n x 1

        # --- solve
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
        x = np.reshape(x.T, [-1, 1])
        equation_right = np.dot(pc.T, x - b)

        exp_para = np.dot(np.linalg.inv(equation_left), equation_right)

        return exp_para

    def __estimateShape(self, x, shapeMU, shapePC, shapeEV, expression, s, R, t2d, lamb):
        '''
        Args:
            x: (2, n). image points (to be fitted)
            shapeMU: (3n, 1)
            shapePC: (3n, n_sp)
            shapeEV: (n_sp, 1)
            expression: (3, n)
            s: scale
            R: (3, 3). rotation matrix
            t2d: (2,). 2d translation
            lambda: regulation coefficient

        Returns:
            shape_para: (n_sp, 1) shape parameters(coefficients)
        '''
        x = x.copy()
        assert(shapeMU.shape[0] == shapePC.shape[0])
        assert(shapeMU.shape[0] == x.shape[1]*3)

        dof = shapePC.shape[1]

        n = x.shape[1]
        sigma = shapeEV
        t2d = np.array(t2d)
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        A = s*P.dot(R)

        # --- calc pc
        pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 199 x n x 3
        pc_3d = np.reshape(pc_3d, [dof*n, 3]) # 199n x 3
        pc_2d = pc_3d.dot(A.T.copy()) # A.T 3 x 2  199 x n x 2

        pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 199

        # --- calc b
        # shapeMU
        mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
        # expression
        exp_3d = expression
        # 
        b = A.dot(mu_3d + exp_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
        b = np.reshape(b.T, [-1, 1]) # 2n x 1

        # --- solve
        equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
        x = np.reshape(x.T, [-1, 1])
        equation_right = np.dot(pc.T, x - b)

        shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

        return shape_para
            
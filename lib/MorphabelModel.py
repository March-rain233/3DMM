import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as transforms
from lib.Model import FitModel as Model
from math import cos, sin
device = 'cpu'

class MorphabelModel:
    def __init__(self, filePath, **kwarg):
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
        self.sp = np.zeros(self.nSP)                            #形状主成分变量
        self.tp = np.zeros(self.nTP)                            #纹理主成分变量
        self.ep = np.zeros(self.nEP)                            #表情主成分变量
        self.shapeMU = self.shapeMU + self.expMU
        self.p = np.eye(3)
        self.t = np.zeros(3)
        self.s = 1

    def transform(self):
        shape = self.shapePC @ self.sp
        exp = self.expPC @ self.ep
        face = self.shapeMU + shape + exp
        face = face.reshape(-1, 3)
        face = face @ self.p.T
        face *= self.s
        face = face + self.t
        tex = ((self.texMU + self.texPC @ self.tp) / 255).reshape(-1, 3).astype(np.float32)
        return face.astype(np.float32), tex.astype(np.float32)
    
    def fit(self, frame, kptPoints):
        '''校准'''
        pass
    
class TraditionalMorphableModel(MorphabelModel):
    def __init__(self, filePath, **kwargs):
        super().__init__(filePath, **kwargs)
        self.maxIter = kwargs['maxIter']  #最大迭代次数

    def fit(self, frame, kptPoints):
        #截取特征点对应的参数
        indices = self.kptInd.astype(np.uint32) * 3
        indices = np.c_[indices, indices + 1, indices + 2].flatten()
        shapeMU = self.shapeMU[indices]
        shapePC = self.shapePC[indices]
        expPC = self.expPC[indices]

        #迭代校准
        for i in range(self.maxIter):
            shape = shapePC @ self.sp

            x3d = (shapeMU + shape + expPC @ self.ep).reshape(-1, 3)
            PA = self.__getPAffine(kptPoints.T, x3d)
            s, R, t = self.__P2SRT(PA)

            self.p[:] = -R[...]
            #self.t[:] = -t[:]

            self.ep = self.__estimateExpression(kptPoints, shapeMU, expPC, self.expEV, shape.reshape(-1, 3).T, s, R, t[:2], 
                                                lamb = 2000).flatten()

            exp = expPC @ self.ep
            self.sp = self.__estimateShape(kptPoints, shapeMU, shapePC, self.shapeEV, exp.reshape(-1, 3).T, s, R, t[:2], 
                                                lamb = 4000).flatten()
    
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
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)  # [2, 3]
        A = s*P.dot(R)  # [2, 3]
        pc = (expPC.T.reshape(-1, 3) @ A.T).reshape(expPC.shape[1], -1).T  # [2 * nver, exp_num]
        b = A @ (shapeMU.reshape(-1, 3).T + shape) + t2d[:, None]  # [2, nver]
        equation_left = pc.T @ pc + lamb * np.diagflat(1/expEV**2)  # [exp_num, exp_num]
        equation_right = pc.T @ (x - b).T.flatten()  # [exp_num,]
        return np.linalg.inv(equation_left) @ equation_right

    def __estimateShape(self, x, shapeMU, shapePC, shapeEV, expression, s, R, t2d, lamb):
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)  # [2, 3]
        A = s*P.dot(R)  # [2, 3]
        pc = (shapePC.T.reshape(-1, 3) @ A.T).reshape(shapePC.shape[1], -1).T  # [2 * nver, shape_num]
        b = A @ (shapeMU.reshape(-1, 3).T + expression) + t2d[:, None]  # [2, nver]
        equation_left = pc.T @ pc + lamb * np.diagflat(1/shapeEV**2)  # [shape_num, shape_num]
        equation_right = pc.T @ (x - b).T.flatten()  # [shape_num,]
        return np.linalg.inv(equation_left) @ equation_right

class MLMorphableModel(MorphabelModel):
    def __init__(self, filePath, **kwarg):
        super().__init__(filePath, **kwarg)
        self.SHAPE_NUM = 40
        self.EXP_NUM = 10
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=127.5, std=128)
        ])
        self.net = Model(self.SHAPE_NUM, self.EXP_NUM)
        self.net.load_state_dict(torch.load(kwarg.get('modelPath', r'model\fit_model.pth')))
        self.net.to(device)
        self.net.eval()

    def fit(self, frame, kptPoints):
        with torch.no_grad():
            kptPoints = kptPoints[:, 
                                  ((0 <= kptPoints[0]) & (kptPoints[0] < frame.shape[1])) &
                                  ((0 <= kptPoints[1]) & (kptPoints[1] < frame.shape[0]))
                                  ]
            frame = self.transforms(frame)[None,...].to(device)
            flms = torch.full([1, 1, frame.shape[2], frame.shape[3]], -1, dtype=torch.float32, device=device)
            flms[0, 0, kptPoints[1], kptPoints[0]] = 1
            input = torch.cat([frame, flms], 1)
            output = self.net(input).cpu()
            self.sp[:self.SHAPE_NUM] = output[0,12:12+self.SHAPE_NUM]
            self.ep[:self.EXP_NUM] = output[0,12+self.SHAPE_NUM:]
            #self.p[:] = output[0,:9].reshape(3, 3)
            #self.t[:] = output[0,9:12]

class TestMorphableModel(MorphabelModel):
    def __init__(self, filePath, **kwarg):
        super().__init__(filePath, **kwarg)
        label = sio.loadmat(kwarg.get('labelPath', r'data\train_data\LFPW\image_test_0002.mat'))
        self.sp[:] = label['Shape_Para'][:,0]
        self.ep[:] = label['Exp_Para'][:,0]
        self.tp[:] = label['Tex_Para'][:,0]
        pose = label['Pose_Para'][0]
        pose[0] *= -1
        self.p[:] = self.angle2matrix(pose[:3])
        #self.t[:] = pose[3:6]
        self.s = pose[6]
    def angle2matrix(self, angles):
        ''' get rotation matrix from three rotation angles(degree). right-handed.
        Args:
            angles: [3,]. x, y, z angles
            x: pitch. positive for looking down.
            y: yaw. positive for looking left. 
            z: roll. positive for tilting head right. 
        Returns:
            R: [3, 3]. rotation matrix.
        '''
        x, y, z = angles[0], angles[1], angles[2]
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

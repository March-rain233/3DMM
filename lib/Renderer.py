from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
import numpy as np

class Renderer:
    def __init__(self, **kwds):
        self.csize = kwds.get('size', (960, 640))       # 画布分辨率
        self.bg = kwds.get('bg', [0.0, 0.0, 0.0])       # 背景色
        self.haxis = kwds.get('haxis', 'y').lower()     # 高度轴
        self.oecs = kwds.get('oecs', [0.0, 0.0, 0.0])   # 视点坐标系ECS原点
        self.near = kwds.get('near', 2.0)               # 相机与视椎体前端面的距离
        self.far = kwds.get('far', 1000.0)              # 相机与视椎体后端面的距离
        self.fovy = kwds.get('fovy', 40.0)              # 相机水平视野角度
        self.dist = kwds.get('dist', 5.0)               # 相机与ECS原点的距离
        self.azim = kwds.get('azim', 0.0)               # 方位角
        self.elev = kwds.get('elev', 0.0)               # 高度角
 
        self.aspect = self.csize[0]/self.csize[1]       # 画布宽高比
        self.cam = None                                 # 相机位置
        self.up = None                                  # 指向相机上方的单位向量
        self._update_cam_and_up()                       # 计算相机位置和指向相机上方的单位向量

        self.left_down = False                          # 左键按下
        self.mouse_pos = (0, 0)                         # 鼠标位置

        # 保存相机初始姿态（视野、方位角、高度角和距离）
        self.home = {'fovy':self.fovy, 'azim':self.azim, 'elev':self.elev, 'dist':self.dist}

        self.mmat = np.eye(4, dtype=np.float32)     # 模型矩阵
        self.vmat = np.eye(4, dtype=np.float32)     # 视点矩阵
        self.pmat = np.eye(4, dtype=np.float32)     # 投影矩阵

        with open(kwds.get('vs', 'shader/VertexShader.glsl'), encoding='utf-8') as f:
            self.vs = f.read()      #顶点着色器代码
        with open(kwds.get('fs', 'shader/FragmentShader.glsl'), encoding='utf-8') as f:
            self.fs = f.read()      #片元着色器代码

        # 设置光照参数
        self.light_dir = np.array([-1, 1, 0], dtype=np.float32)     # 光线照射方向
        self.light_color = np.array([1, 1, 1], dtype=np.float32)    # 光线颜色
        self.ambient = np.array([0.2, 0.2, 0.2], dtype=np.float32)  # 环境光颜色
        self.shiny = 50                                             # 高光系数
        self.specular = 1.0                                         # 镜面反射系数
        self.diffuse = 0.7                                          # 漫反射系数
        self.pellucid = 0.5                                         # 透光度

    def draw(self):
        glUseProgram(self.program)
    
        loc = glGetAttribLocation(self.program, 'a_Position')
        self.vertices.bind()
        glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*4, self.vertices)
        glEnableVertexAttribArray(loc)
        self.vertices.unbind()
     
        loc = glGetAttribLocation(self.program, 'a_Normal')
        self.normal.bind()
        glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*4, self.normal)
        glEnableVertexAttribArray(loc)
        self.normal.unbind()

        loc = glGetAttribLocation(self.program, 'a_Color')
        self.colors.bind()
        glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*4, self.colors)
        glEnableVertexAttribArray(loc)
        self.colors.unbind()

        loc = glGetUniformLocation(self.program, 'u_ProjMatrix')
        glUniformMatrix4fv(loc, 1, GL_FALSE, self.get_pmat(), None)

        loc = glGetUniformLocation(self.program, 'u_ViewMatrix')
        glUniformMatrix4fv(loc, 1, GL_FALSE, self.get_vmat(), None)

        loc = glGetUniformLocation(self.program, 'u_ModelMatrix')
        glUniformMatrix4fv(loc, 1, GL_FALSE, self.mmat, None)

        loc = glGetUniformLocation(self.program, 'u_CamPos')
        glUniform3f(loc, *self.cam)

        loc = glGetUniformLocation(self.program, 'u_LightDir')
        glUniform3f(loc, *self.light_dir)

        loc = glGetUniformLocation(self.program, 'u_LightColor')
        glUniform3f(loc, *self.light_color)

        loc = glGetUniformLocation(self.program, 'u_AmbientColor')
        glUniform3f(loc, *self.ambient)

        loc = glGetUniformLocation(self.program, 'u_Shiny')
        glUniform1f(loc, self.shiny)

        loc = glGetUniformLocation(self.program, 'u_Specular')
        glUniform1f(loc, self.specular)

        loc = glGetUniformLocation(self.program, 'u_Diffuse')
        glUniform1f(loc, self.diffuse)

        loc = glGetUniformLocation(self.program, 'u_Pellucid')
        glUniform1f(loc, self.pellucid)

        self.indices.bind()
        glDrawElements(GL_TRIANGLES, self.n, GL_UNSIGNED_INT, None)
        self.indices.unbind()

        glUseProgram(0)

    def render(self):
        """重绘事件函数"""

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # 清除屏幕及深度缓存
        self.draw() # 绘制模型
        glutSwapBuffers() # 交换缓冲区


    def prepare(self):
        vshader = shaders.compileShader(self.vs, GL_VERTEX_SHADER)
        fshader = shaders.compileShader(self.fs, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vshader, fshader)


    def show(self):
        glutInit() # 初始化glut库

        sw, sh = glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT)
        left, top = (sw-self.csize[0])//2, (sh-self.csize[1])//2

        glutInitWindowSize(*self.csize) # 设置窗口大小
        glutInitWindowPosition(left, top) # 设置窗口位置
        glutCreateWindow('Data View Toolkit') # 创建窗口
        
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH) # 设置显示模式
        glClearColor(*self.bg, 1.0) # 设置背景色
        glEnable(GL_DEPTH_TEST) # 开启深度测试
        glDepthFunc(GL_LEQUAL) # 设置深度测试函数的参数

        self.prepare() # GL初始化后、开始绘制前的预处理

        glutDisplayFunc(self.render) # 绑定重绘事件函数
        glutReshapeFunc(self.reshape) # 绑定窗口大小改变事件函数
        glutMouseFunc(self.click) # 绑定鼠标按键和滚轮事件函数
        glutMotionFunc(self.drag) # 绑定鼠标拖拽事件函数
        
        glutMainLoop() # 进入glut主循环

    def SetModel(self, vertices, colors, indices):
        self.vertices = vbo.VBO(vertices)
        self.colors = vbo.VBO(colors)
        self.indices = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)

        primitive = vertices[indices]
        a = primitive[::3]
        b = primitive[1::3]
        c = primitive[2::3]
        normal = np.repeat(np.cross(b-a, c-a), 3, axis=0)

        temp = np.full_like(vertices, 0.00001)
        for i, ind in enumerate(indices):
            temp[ind] += normal[i]
        temp /= np.linalg.norm(temp, axis=1)[...,None]
        self.normal = vbo.VBO(temp)
        
        self.n = len(indices)

    def get_vmat(self):
        """返回视点矩阵"""
 
        camX, camY, camZ = self.cam
        oecsX, oecsY, oecsZ = self.oecs
        upX, upY, upZ = self.up
 
        f = np.array([oecsX-camX, oecsY-camY, oecsZ-camZ], dtype=np.float64)
        f /= np.linalg.norm(f)
        s = np.array([f[1]*upZ - f[2]*upY, f[2]*upX - f[0]*upZ, f[0]*upY - f[1]*upX], dtype=np.float64)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
 
        return np.array([
            [s[0], u[0], -f[0], 0],
            [s[1], u[1], -f[1], 0],
            [s[2], u[2], -f[2], 0],
            [- s[0]*camX - s[1]*camY - s[2]*camZ, 
            - u[0]*camX - u[1]*camY - u[2]*camZ, 
            f[0]*camX + f[1]*camY + f[2]*camZ, 1]
        ], dtype=np.float32)

    def get_pmat(self):
        """返回投影矩阵"""
 
        right = np.tan(np.radians(self.fovy/2)) * self.near
        left = -right
        top = right/self.aspect
        bottom = left/self.aspect
        rw, rh, rd = 1/(right-left), 1/(top-bottom), 1/(self.far-self.near)
 
        return np.array([
            [2 * self.near * rw, 0, 0, 0],
            [0, 2 * self.near * rh, 0, 0],
            [(right+left) * rw, (top+bottom) * rh, -(self.far+self.near) * rd, -1],
            [0, 0, -2 * self.near * self.far * rd, 0]
        ], dtype=np.float32)

    def _update_cam_and_up(self, oecs=None, dist=None, azim=None, elev=None):
        """根据当前ECS原点位置、距离、方位角、仰角等参数，重新计算相机位置和up向量"""

        if not oecs is None:
            self.oecs = [*oecs,]
 
        if not dist is None:
            self.dist = dist
 
        if not azim is None:
            self.azim = (azim+180)%360 - 180
 
        if not elev is None:
            self.elev = (elev+180)%360 - 180
 
        up = 1.0 if -90 <= self.elev <= 90 else -1.0
        azim, elev  = np.radians(self.azim), np.radians(self.elev)
        d = self.dist * np.cos(elev)

        if self.haxis == 'z':
            azim -= 0.5 * np.pi
            self.cam = [d*np.cos(azim)+self.oecs[0], d*np.sin(azim)+self.oecs[1], self.dist*np.sin(elev)+self.oecs[2]]
            self.up = [0.0, 0.0, up]
        else:
            self.cam = [d*np.sin(azim)+self.oecs[0], self.dist*np.sin(elev)+self.oecs[1], d*np.cos(azim)+self.oecs[2]]
            self.up = [0.0, up, 0.0]

    def reshape(self, w, h):
        """改变窗口大小事件函数"""
 
        self.csize = (w, h)
        self.aspect = self.csize[0]/self.csize[1] if self.csize[1] > 0 else 1e4
        glViewport(0, 0, self.csize[0], self.csize[1])
 
        glutPostRedisplay()

    def click(self, btn, state, x, y):
        """鼠标按键和滚轮事件函数"""
 
        if btn == 0: # 左键
            if state == 0: # 按下
                self.left_down = True
                self.mouse_pos = (x, y)
            else: # 弹起
                self.left_down = False 
        elif btn == 2 and state ==1: # 右键弹起，恢复相机初始姿态
            self._update_cam_and_up(dist=self.home['dist'], azim=self.home['azim'], elev=self.home['elev'])
            self.fovy = self.home['fovy']
        elif btn == 3 and state == 0: # 滚轮前滚
            self.fovy *= 0.95
        elif btn == 4 and state == 0: # 滚轮后滚
            self.fovy += (180 - self.fovy) / 180
        
        glutPostRedisplay()

    def drag(self, x, y):
        """鼠标拖拽事件函数"""
        
        dx, dy = x - self.mouse_pos[0], y - self.mouse_pos[1]
        self.mouse_pos = (x, y)

        azim = self.azim - (180*dx/self.csize[0]) * (self.up[2] if self.haxis == 'z' else self.up[1])
        elev = self.elev + 90*dy/self.csize[1]
        self._update_cam_and_up(azim=azim, elev=elev)
        
        glutPostRedisplay()

if __name__=='__main__':
    r = Renderer()
    vs = np.array([[0, 1, 0], [-1, -1, 0], [1, -1, 0], [0,0,1]], dtype=np.float32)
    colos = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.333,0.333,0.333]], dtype=np.float32)
    indices = np.array([0,1,2, 0,2,3, 1,0,3, 2,1,3], dtype=np.int32)
    r.SetModel(vs, colos, indices)
    r.show()
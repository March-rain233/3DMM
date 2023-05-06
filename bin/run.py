import cv2
import dlib
import numpy as np
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
from lib.MorphabelModel import MorphabelModel as Model
from lib.Renderer import Renderer

def main():
    m = Model('model\model2019_fullHead.h5')
    v, c = m.transform()
    scale = np.max(np.max(v, axis=0)-np.min(v, axis=0))
    v/=scale
    r = Renderer()
    r.SetModel(v, c, m.tl.flatten())
    r.show()
    # cap = cv2.VideoCapture(0) 
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    # while(cap.isOpened()):
    # # 获取一帧
    #     ret, frame = cap.read()

    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = detector(gray, 1)
    #     for face in faces:

    #         # # 8.1 绘制矩形框
    #         # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3)
    #         # 8.2 检测关键点
    #         shape = predictor(gray, face)
    #         #8.3 获取关键点坐标
    #         for pt in shape.parts():
    #             # 每个点的坐标
    #             pt_position = (pt.x, pt.y)
    #             # 绘制关键点
    #             cv2.circle(frame, pt_position, 3, (255, 0, 0), -1)

    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__=='__main__':
    main()
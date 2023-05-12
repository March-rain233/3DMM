import cv2
import dlib
import numpy as np
import sys
import time
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
from lib.MorphabelModel import MorphabelModel as Model
from lib.Renderer import Renderer

n = [i for i in range(0, 68, 1)]

def fitFromPicture(filepath, model):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    frame = cv2.imread(filepath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    shape = predictor(gray, faces[0])
    points = np.array([[p.x, frame.shape[1] - p.y] for p in shape.parts()])
    
    start = time.time()
    model.fit(points)
    end = time.time()
    print(f'耗时{end - start}s')

    for i in n:
        cv2.circle(frame, (points[i, 0], frame.shape[1] - points[i, 1]), 3, (255, 0, 0), -1)

    v, c = model.transform()
    indices = model.tri.flatten()
    scale = 2.5/np.max(np.max(v, axis=0)-np.min(v, axis=0))
    v*=scale
    c[model.kptInd[n]] = [0,0,1]

    r = Renderer()
    r.setModel(v, c, indices)
    r.setCompare(frame[...,::-1])
    r.show()

def fitFromCapture(model):
    r = Renderer()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0) 
    def getModle():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces):
            start = time.time()
            shape = predictor(gray, faces[0])
            points = np.array([[p.x, frame.shape[1] - p.y] for p in shape.parts()])
            model.fit(points)
            end = time.time()
            #print(f'耗时{end - start}s')
            for i in n:
                cv2.circle(frame, (points[i, 0], frame.shape[1] - points[i, 1]), 3, (255, 0, 0), -1)

            v, c = model.transform()
            indices = model.tri.flatten()
            scale = 2.5/np.max(np.max(v, axis=0)-np.min(v, axis=0))
            v*=scale
            c[model.kptInd[n]] = [0,0,1]
            r.setModel(v, c, indices)
            r.setCompare(frame[...,::-1])
    r.beforeDraw = getModle
    r.show()
    cap.release()

def main():
    m = Model('model\BFM.mat', 50)
    fitFromPicture('data/face1.jpg', m)
    #fitFromCapture(m)

if __name__=='__main__':
    main()
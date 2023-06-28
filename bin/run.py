import cv2
import dlib
import numpy as np
import sys
import time
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
#from lib.MorphabelModel import MLMorphableModel as Model
from lib.MorphabelModel import TraditionalMorphableModel as Model
#from lib.MorphabelModel import TestMorphableModel as Model
from lib.Renderer import Renderer

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
STD_SIZE = 120

def processFrame(frame, face):
    shape = PREDICTOR(frame, face)
    keyPoints = np.array([[p.x, p.y] for p in shape.parts()]).T
    detail = dlib.get_face_chip_details(shape, size = STD_SIZE, padding=0.5)
    clip = dlib.extract_image_chip(frame, detail)
    clipFace = DETECTOR(clip, 1)[0]
    clipShape = PREDICTOR(clip, clipFace)
    clipKeyPoints = np.array([[p.x, p.y] for p in clipShape.parts()]).T
    #clip, clipKeyPoints = frame, keyPoints
    return clip, clipKeyPoints, keyPoints

def fitFromPicture(filepath, model):
    r = Renderer()
    frame = cv2.imread(filepath)
    frame = cv2.resize(frame, None, fx=2, fy=2)
    faces = DETECTOR(frame, 1)
    if len(faces):
        clip, clipKeyPoints, keyPoints = processFrame(frame, faces[0]) 
        frame, keyPoints = clip, clipKeyPoints
        start = time.time()
        model.fit(clip, clipKeyPoints)
        end = time.time()
        print(f'耗时{end - start}s')

        for i in range(keyPoints.shape[1]):
            cv2.circle(frame, (keyPoints[0, i], keyPoints[1, i]), 1, (255, 0, 0), -1)

        v, c = model.transform()
        c[:] = 1
        indices = model.tri.flatten()
        scale = 2.5/np.max(np.max(v, axis=0)-np.min(v, axis=0))
        v*=scale
        c[model.kptInd[:]] = [0,0,1]

        r.setModel(v, c, indices)

    r.setCompare(frame[:,::-1,::-1])
    r.show()

def fitFromCapture(model):
    r = Renderer()

    cap = cv2.VideoCapture(0) 
    def getModle():
        ret, frame = cap.read()
        faces = DETECTOR(frame, 1)
        if len(faces):
            clip, clipKeyPoints, keyPoints = processFrame(frame, faces[0]) 
            frame = clip
            keyPoints = clipKeyPoints
            start = time.time()
            model.fit(clip, clipKeyPoints)
            end = time.time()
            #print(f'耗时{end - start}s')
            for i in range(keyPoints.shape[1]):
                cv2.circle(frame, (keyPoints[0, i], keyPoints[1, i]), 1, (255, 0, 0), -1)

            v, c = model.transform()
            c[:] = 1
            indices = model.tri.flatten()
            scale = 2.5/np.max(np.max(v, axis=0)-np.min(v, axis=0))
            v*=scale
            #c[model.kptInd[:]] = [0,0,1]
            r.setModel(v, c, indices)

        r.setCompare(np.ascontiguousarray(frame[:,::-1,::-1]))
    r.beforeDraw = getModle
    r.show()
    cap.release()

def main():
    m = Model(
        filePath=r'model\BFM.mat', 
        maxIter=3, 
        modelPath=r'model/fit_model.pth',
        labelPath=r'data\300W_LP\LFPW\LFPW_image_train_0645_1.mat'
    )
    #fitFromPicture(r'data\300W_LP\LFPW\LFPW_image_train_0645_1.jpg', m)
    fitFromCapture(m)

if __name__=='__main__':
    main()
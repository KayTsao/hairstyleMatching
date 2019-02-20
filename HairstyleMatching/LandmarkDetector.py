import cv2
import dlib
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
img = cv2.imread('test5.png')
faces = detector(img,1)
if (len(faces) > 0):
    for k,d in enumerate(faces):
        #cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
        shape = landmark_predictor(img,d)
        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y),5,(0,255,0), -1, 8)
            if(i==19 or i==24 or i== 8 or i==30):
                print(i, shape.part(i).x, shape.part(i).y)
                cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255))
#img = cv2.resize(img,())
cv2.imwrite("./lmk.png", img)
#cv2.waitKey(0)

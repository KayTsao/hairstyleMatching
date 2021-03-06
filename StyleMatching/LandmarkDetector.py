import cv2
import dlib

def KPDtc(src_img):
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('StyleMatching/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(src_img)
    faces = detector(img,1)
    
    if (len(faces) > 0):
        for k,d in enumerate(faces):
            #cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
            shape = landmark_predictor(img,d)
            #for i in range(68):
                #cv2.circle(img, (shape.part(i).x, shape.part(i).y),5,(0,255,0), -1, 8)
                #if(i==19 or i==24 or i== 8 or i==30):
                    #cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255))
            
    #out_dir = 'test1.png' #'{}_lmk.png'.format(src_img)
    #cv2.imwrite(out_dir, img)
    return shape.part(19), shape.part(24), shape.part(30), shape.part(8)
    #img = cv2.resize(img,())
    '''
    
    
    #cv2.waitKey(0)
    p_eye1.x = shape.part(19).x
    p_eye1.y = shape.part(19).y
    p_eye2.x = shape.part(24).x
    p_eye2.y = shape.part(24).y
    p_nose.x = shape.part(8).x
    p_nose.y = shape.part(8).y
    p_chin.x = shape.part(30).x
    p_chin.y = shape.part(30).y
    
    '''

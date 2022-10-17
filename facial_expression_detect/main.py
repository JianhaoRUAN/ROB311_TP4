# 1  
import cv2
import dlib
import joblib
from KNN_classifier import detect_expression
 
# 2  
def plot_rectangle(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255,0,0), 4)
    return image
 
def main():
    # define facial expression
    dict_expression = {"angry" : 0 , "disgust" : 1 , "fear" : 2 , "happy" : 3 , "neutral" : 4 , "sad" : 5 , "surprise" : 6}

    #load model
    neigh = joblib.load('neigh.pkl')

    # 3 open cam
    capture = cv2.VideoCapture(0)
    # 4 
    if capture.isOpened() is False:
        print("Camera Error !")
    # 5 read frame
    while True:
        ret, frame = capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # BGR to GRAY
            gray = cv2.resize(gray, (int(1920/4), int(1080/4))) 
 
            # 6 use detector in dlib to extract face
            detector = dlib.get_frontal_face_detector()
            det_result = detector(gray, 1)  # detect and get result
            for face in det_result:
                face_img = gray[ face.top() : face.bottom(), face.left():face.right()]
                if face.top()  > 0 :
                    face_img = cv2.resize(face_img, (48, 48)) 
                    cv2.imshow("face detection with dlib",face_img)
                    
                    expression = detect_expression(face_img, neigh)
                    expression_str = list(dict_expression.keys())[list(dict_expression.values()).index(expression)]
                    print(expression_str)
            








            # 7 draw rectangle
            # dets_image = plot_rectangle(frame, det_result)  #   
 
            # 8  
            #cv2.imshow("face detection with dlib", dets_image)
 
            # 9  "ESC"，to exit
            if cv2.waitKey(1) == 27:
                break
 
    # 10 release
    capture.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()
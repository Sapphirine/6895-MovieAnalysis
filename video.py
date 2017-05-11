#Import required modules
import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.externals import joblib
import numpy as np
import cv2

emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

cap = cv2.VideoCapture('Downfall.mp4')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

# Face detector
detector = dlib.get_frontal_face_detector()
#Landmark identifier. Set the filename to whatever you named the downloaded file
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# load facial expression model SVM
clf2 = joblib.load('SVC_model.pkl')


while(cap.isOpened()):
    ret, frame = cap.read()

    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        #Detect the faces in the image
        detections = detector(clahe_image, 1)

        #For each detected face
        for k,d in enumerate(detections):
            #Get coordinates
            shape = predictor(clahe_image, d)
            xlist, ylist = [], []
            #Store X and Y coordinates in two lists
            for i in range(1,68):
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
                #For each point, draw a red circle with thickness2 on the original frame
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)

            #Get the mean of both axes to determine centre of gravity
            xmean, ymean = np.mean(xlist), np.mean(ylist)
            #get distance between each point and the central point in both axes
            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]

            #If x-coordinates of the set are the same, the angle is 0,
            #catch to prevent 'divide by 0' error in function
            if xlist[26] == xlist[29]:
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)
            if anglenose < 0:
                anglenose += 90
            else:
                anglenose -= 90

            landmarks_vectorised = []
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append(anglerelative)

            ind = int(clf2.predict([landmarks_vectorised]))
            #show facial expression on the screen
            cv2.putText(frame, emotions[ind], (25, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), thickness=1)

        # write the frame
        out.write(frame)
        #cv2.imshow("image", frame) #Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


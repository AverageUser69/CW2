import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'StudentImages'
images = []
classNames = []
listof = os.listdir(path)
print(listof)
for cl in listof:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def searchencoded(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def marking(stdid):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if stdid not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{stdid},{dtString}')


encodeListKnown = searchencoded(images)
print('Encoding Images')
cap = cv2.VideoCapture(0)
width, height = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


while True:
    success, img = cap.read()
    # img = captureScreen()
    resizedimage = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    resizedimage = cv2.cvtColor(resizedimage, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(resizedimage)
    encodesCurFrame = face_recognition.face_encodings(resizedimage, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            stdid = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, stdid, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            marking(stdid)

    cv2.imshow('Attendance', img)
    cv2.waitKey(1)
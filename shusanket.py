import cv2
import mediapipe as mp


capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
face_cascade = cv2.CascadeClassifier('haar_face.xml')
while True:
    istrue, shus = capture.read()
    frame = cv2.resize(shus, (1100, 800), interpolation=cv2.INTER_AREA)
    colorgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(colorgb)




    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.putText(frame,"SHUSANKET BASYAL", (x,y-80), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (170,255,0),2)
        cv2.putText(frame, "AGE:20", (x, y - 30), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (170,255,0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handif in result.multi_hand_landmarks:
            for id, lm in enumerate(handif.landmark):
                print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)



            mpDraw.draw_landmarks(frame, handif, mpHands.HAND_CONNECTIONS)



    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime







    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('s'):
        break

capture.release()
cv2.destroyAllWindows()









import cv2
import mediapipe as mp
hand_mpHands = mp.solutions.hands
hand_hands = hand_mpHands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)
hand_mpDraw = mp.solutions.drawing_utils
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec_dots = mp_drawing.DrawingSpec(color = (201,194,2),thickness=1, circle_radius=2)
drawing_spec_line = mp_drawing.DrawingSpec(color = (255,255,255),thickness=2, circle_radius=1)
cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=1,min_tracking_confidence=0.5)
while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    hand_results = hand_hands.process(image)
    results = face_mesh.process(image)
    print(results.multi_face_landmarks)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                tips = [0,4,8,12,16,20]
                if id in tips:
                    cv2.circle(image,(cx,cy),15,(255,255,255),cv2.FILLED)
            hand_mpDraw.draw_landmarks(
                image,
                handLms,
                hand_mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec =  hand_mpDraw.DrawingSpec(color=(0,0,0)),
                connection_drawing_spec =  hand_mpDraw.DrawingSpec(color=(201, 194, 2))
            )
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec_dots,
            connection_drawing_spec=drawing_spec_line)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()

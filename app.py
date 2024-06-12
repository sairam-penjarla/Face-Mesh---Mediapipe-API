import cv2
import mediapipe as mp
import yaml

class HandFaceMeshApp:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.cap = cv2.VideoCapture(self.config['video_source'])
        self.hand_mpHands = mp.solutions.hands
        self.hand_hands = self.hand_mpHands.Hands(
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence'])
        self.hand_mpDraw = mp.solutions.drawing_utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hand_drawing_spec = self.hand_mpDraw.DrawingSpec(color=(0,0,0)),
        self. connection_drawing_spec = self.hand_mpDraw.DrawingSpec(color=(201, 194, 2))
        self.drawing_spec_dots = self.mp_drawing.DrawingSpec(color = (201,194,2),thickness=1, circle_radius=2)
        self.drawing_spec_line = self.mp_drawing.DrawingSpec(color = (255,255,255),thickness=2, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=self.config['face_min_detection_confidence'],
            min_tracking_confidence=self.config['face_min_tracking_confidence'])

    def process_frame(self, image):
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Process hand landmarks
        hand_results = self.hand_hands.process(image)
        # Process face landmarks
        face_results = self.face_mesh.process(image)
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for handLms in hand_results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    h,w,c = image.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    tips = [0,4,8,12,16,20]
                    if id in tips:
                        cv2.circle(image,(cx,cy),15,(255,255,255),cv2.FILLED)
                self.hand_mpDraw.draw_landmarks(
                    image,
                    handLms,
                    self.hand_mpHands.HAND_CONNECTIONS,
                    landmark_drawing_spec =  self.hand_mpDraw.DrawingSpec(color=(0,0,0)),
                    connection_drawing_spec =  self.hand_mpDraw.DrawingSpec(color=(201, 194, 2))
                )
        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec_dots,
                    connection_drawing_spec=self.drawing_spec_line)
        return image

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            # Process the frame
            image = self.process_frame(image)
            # Display the frame
            cv2.imshow('Hand and Face Mesh', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandFaceMeshApp('config.yaml')
    app.run()

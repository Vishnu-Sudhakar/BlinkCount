import cv2
import dlib
import os
from scipy.spatial import distance as dist

# Get the path to the shape predictor file
script_dir = os.path.dirname(os.path.realpath(__file__))
shape_predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

# Load facial landmark predictor
predictor = dlib.shape_predictor(shape_predictor_path)
detector = dlib.get_frontal_face_detector()

# Function to calculate EAR
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Blink detection threshold
EAR_THRESHOLD = 0.25

blink_count = 0
blink_flag = False

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        # Extract left and right eye coordinates
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Calculate EAR for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the threshold
        if ear < EAR_THRESHOLD:
            if not blink_flag:
                blink_count += 1
                blink_flag = True
        else:
            blink_flag = False

    # Display the blink count on the frame
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Blink Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

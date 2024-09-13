import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error: Unable to read frame from camera.")
                break

            # Convert the BGR image to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw pose landmarks.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Convert the frame to JPEG format.
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Unable to encode frame as JPEG.")
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Destroy any OpenCV windows (if any were opened)

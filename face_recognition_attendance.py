# Import required libraries
import fileinput
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
# libraries for blink detection
import dlib
from scipy.spatial import distance as dist
import requests  # For sending the attendance file via Discord Webhook
from playsound import playsound  # For playing the audio files

def play_sound_file(sound_file='short_success.mp3'):
    ''' playing the audio file '''
    playsound(sound_file)

# initialize the frame counters ( how many  consec frames the eye is closed )
COUNTER = 0


def load_known_faces(photos_path="photos"):
    """Load face encodings and names from a folder of images."""
    known_face_encodings = []
    known_faces_names = []

    for filename in os.listdir(photos_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(photos_path, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                name = os.path.splitext(filename)[0]
                known_faces_names.append(name)

    return known_face_encodings, known_faces_names


def eye_aspect_ratio(eye : list[int]):
    ''' computes the ear of each eye individually by taking the coordinates of the six points of the eye '''
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def initialize_dlib():
    ''' statrting detector functions '''
    print("[INFO] loading facial landmark predictor...")

    # this func returns the bounding box coordinates for each face (left,top,right,bottom)
    detector = dlib.get_frontal_face_detector()

    # this func returns the 68 facial landmarks which contains 12 eye landmarks
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor


def create_attendance_file():
    """Create a CSV file for today's attendance and return the writer."""
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    filename = current_date + ".csv"
    file = open(current_date + ".csv", "w+", newline='')
    writer = csv.writer(file)
    return file, writer, filename


# initialize the detector functions
detector, predictor = initialize_dlib()


def check_blink(frame):
    global COUNTER
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray_frame, 0)
    for face in faces:

        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray_frame, face)

        # (part(i)) is a methode that return the specific landmark by index i
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(0, 68)]

        # extract the left and right eye coordinates
        leftEye = shape[42:48]
        rightEye = shape[36:42]

        # compute the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # this checks if the eye is closed it will increas the counter and will return false because it still isn't a blink
        if ear < 0.2:
            COUNTER += 1
            return False

        # if the code reach hear this means that the eye is opend and will check does the eye closed for specific number of frames ,if not then it is just a wrong measure
        else:
            if COUNTER > 5:
                COUNTER = 0
                return True
            COUNTER = 0
            return False


def process_frame(frame, known_face_encodings, known_faces_names, students, writer):
    """Process one frame: detect faces, recognize them, and mark attendance."""
    now = datetime.now()

    # Resize frame to 1/4 size for performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""

        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in students:
            students.remove(name)
            print(name, "is present")
            current_time = now.strftime("%H-%M-%S")
            writer.writerow([name, current_time])
            play_sound_file()

    return face_names


def send_discord_file(file_path):
    """Send the attendance CSV file to Discord using a webhook."""
    webhook_url = "YOUR_WEBHOOK_URL"  # Replace with your Discord webhook URL
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(webhook_url, files=files)

    if response.status_code == 204:
        print("CSV file successfully sent to Discord.")
    else:
        print("Failed to send file:", response.text)


def run_attendance_system():
    """Main loop for face recognition attendance system."""
    # Load known faces
    known_face_encodings, known_faces_names = load_known_faces()
    students = known_faces_names.copy()

    # Create attendance file
    file, writer, filename = create_attendance_file()

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if check_blink(frame):
            # Process frame and update attendance
            process_frame(frame, known_face_encodings, known_faces_names, students, writer)

        # Display the video feed
        cv2.imshow("Attendance System", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    file.close()
    # Send the attendance file to Discord after closing the system
    send_discord_file(filename)


if __name__ == "__main__":
    run_attendance_system()

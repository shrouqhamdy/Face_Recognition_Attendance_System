# Import required libraries
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


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


def create_attendance_file():
    """Create a CSV file for today's attendance and return the writer."""
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    file = open(current_date + ".csv", "w+", newline='')
    writer = csv.writer(file)
    return file, writer


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

    return face_names


def run_attendance_system():
    """Main loop for face recognition attendance system."""
    # Load known faces
    known_face_encodings, known_faces_names = load_known_faces("../../Desktop/new file/photos")
    students = known_faces_names.copy()

    # Create attendance file
    file, writer = create_attendance_file()

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

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


if __name__ == "__main__":
    run_attendance_system()

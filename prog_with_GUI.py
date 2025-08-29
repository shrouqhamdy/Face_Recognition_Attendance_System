# Import required libraries
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# libraries for blink detection
import dlib
from scipy.spatial import distance as dist
import requests
from playsound import playsound

#GUI libraries
import tkinter as tk
from tkinter import ttk, messagebox, Tk
from PIL import Image, ImageTk

# Global variables
root = tk.Tk()
table = None
subject_entry = None
start_time_entry = None
lecture_count_label = None
attendee_count_label = None
current_lecture_label = None
camera_label = None

schedule = []
current_lecture_index = 0
lecture_count = 0
attendee_count = 0

video_capture = None
known_face_encodings = []
known_faces_names = []
students = []
writer = None
attendance_file = None
timer_id = None
COUNTER = 0
frame_count = 0


# ---------------- Functions ----------------

def setup_gui():
    global subject_entry, start_time_entry, lecture_count_label, attendee_count_label, current_lecture_label, table, camera_label

    root.title("Attendance System")
    root.geometry("700x400")
    root.resizable(False, False)

    # --- Right side inputs ---
    tk.Label(root, text="Lecture Name:",
             font=("Arial", 9),
             width=16,
             height=1,
             relief="solid",
             bd=0,
             highlightbackground="#8C75FF",
             highlightthickness=1).place(x=30, y=30)

    subject_entry = tk.Entry(root, width=20, font=("Arial", 9))
    subject_entry.place(x=160, y=30)

    tk.Label(root, text="Start Time (HH:MM):",
             font=("Arial", 9),
             width=16,
             height=1,
             relief="solid",
             bd=0,
             highlightbackground="#8C75FF",
             highlightthickness=1).place(x=30, y=60)

    start_time_entry = tk.Entry(root, width=20, font=("Arial", 9))
    start_time_entry.place(x=160, y=60)

    # Buttons
    tk.Button(root, text="Add Lecture",
              font=("Arial", 9),
              bg="#7155FF",
              fg="white",
              command=add_lecture).place(x=115, y=95)

    lecture_count_label = tk.Label(root, text="Lectures Today: 0")
    lecture_count_label.place(x=30, y=150)

    attendee_count_label = tk.Label(root, text="Attendees Today: 0")
    attendee_count_label.place(x=30, y=180)

    current_lecture_label = tk.Label(root, text="Current Lecture: None")
    current_lecture_label.place(x=100, y=220)

    tk.Button(root, text="Start Day", font=("Arial", 9),
              bg="#7155FF", fg="white", command=start_day).place(x=45, y=278, width=80)

    tk.Button(root, text="End Lecture", font=("Arial", 9),
              bg="#7155FF", fg="white", command=end_lecture).place(x=185, y=278, width=80)

    tk.Button(root, text="End Day & Send", font=("Arial", 9),
              bg="#7155FF", fg="white", command=end_day).place(x=105, y=325, width=100)


    # --- Left side: Table for schedule ---
    table_frame = tk.Frame(root,width=300,height=350)
    table_frame.place(x=350, y=25)
    table_frame.pack_propagate(False)

    table = ttk.Treeview(table_frame, columns=("Subject", "Start Time"), show="headings")
    table.heading("Subject", text="Lecture Name")
    table.heading("Start Time", text="Start Time")
    table.column("Subject", width=150,)
    table.column("Start Time", width=150)
    table.pack(fill="both", expand=True)

    camera_label = tk.Label(root)
    camera_label.place(x=350, y=25,width=300,height=350)
    camera_label.lower()


def add_lecture():
    global lecture_count
    subject = subject_entry.get().strip()
    start_time_str = start_time_entry.get().strip()

    if not subject:
        messagebox.showwarning("Input required", "Please enter a lecture name")
        return
    if not start_time_str:
        messagebox.showwarning("Input required", "Please enter start time in HH:MM")
        return

    try:
        start_time = datetime.strptime(start_time_str, "%H:%M").time()
    except ValueError:
        messagebox.showerror("Error", "Start time must be in HH:MM format (14:30)")
        return

    for lecture in schedule:
        if lecture['start_time'] == start_time:
            messagebox.showerror("Duplicate Time", f"A lecture is already scheduled at {start_time.strftime('%H:%M')}")
            return

    schedule.append({'subject': subject, 'start_time': start_time})
    schedule.sort(key=lambda x: x['start_time'])

    table.delete(*table.get_children())
    for lecture in schedule:
        table.insert("", "end", values=(lecture['subject'], lecture['start_time'].strftime("%H:%M")))

    # Update lecture counter
    lecture_count = len(schedule)
    lecture_count_label.config(text=f"Lectures Today: {lecture_count}")

    # Clear the entry
    subject_entry.delete(0, tk.END)
    start_time_entry.delete(0, tk.END)
    check_schedule()


def check_schedule():
    """Check the schedule every second and start the lecture when its time comes."""
    global current_lecture_index, schedule, root, timer_id

    if current_lecture_index >= len(schedule):
        print("All lectures completed.")
        return

    current_time = datetime.now().time()
    current_lecture = schedule[current_lecture_index]

    if current_time >= current_lecture['start_time']:
        start_lecture()
        current_lecture_index += 1

    # Repeat every second
    timer_id = root.after(1000, check_schedule)


def start_lecture():
    global current_lecture_index, current_lecture_label, table, camera_label, video_capture
    if current_lecture_index >= len(schedule):
        print("No more lectures to start.")
        return
    current = schedule[current_lecture_index]
    current_lecture_label.config(text=f"Current Lecture: {current['subject']}")

    table.pack_forget()
    camera_label.lift()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        messagebox.showerror("Error", "Failed to open camera")
        return
    run_attendance_system()


def update_camera_frame(frame):
    global camera_label
    if frame is None:
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.config(image=imgtk)


def end_lecture():
    global video_capture, camera_label, table, writer, attendance_file, lecture_count

    if video_capture and video_capture.isOpened():
        video_capture.release()
        video_capture = None
    if writer and attendance_file:
        attendance_file.close()
        writer = None
        attendance_file = None
        # Remove the current lecture from schedule
        if current_lecture_index <= len(schedule):
            schedule.pop(current_lecture_index)
            lecture_count = len(schedule)
            lecture_count_label.config(text=f"Lectures Today: {lecture_count}")

            # Update the table
            table.delete(*table.get_children())
            for lecture in schedule:
                table.insert("", "end", values=(lecture['subject'], lecture['start_time'].strftime("%H:%M")))
    cv2.destroyAllWindows()
    camera_label.lower()
    table.pack(fill="both", expand=True)
    print("Lecture ended.")
    check_schedule()


def start_day():
    # fill it
    print("Starting the day...")
    check_schedule()


def end_day():
    #fill it
    print("Ending the day & sending attendance...")


def play_sound_file(sound_file='short_success.mp3'):
    playsound(sound_file)


def load_known_faces(photos_path="photos",):

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



def eye_aspect_ratio(eye: list[int]):
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
detector, predictor = initialize_dlib()# initialize the detector functions


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
    global video_capture, camera_label, known_face_encodings, known_faces_names, students, writer, attendance_file
    if video_capture is None or not video_capture.isOpened():
        print("Camera not initialized or failed to open")
        return
    #load known faces
    known_face_encodings, known_faces_names = load_known_faces()
    students = known_faces_names.copy()
    # Create attendance file
    attendance_file, writer, filename = create_attendance_file()

    while video_capture and video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            break
        if check_blink(frame):
            # Process frame and update attendance
            process_frame(frame, known_face_encodings, known_faces_names, students, writer)
        update_camera_frame(frame)
        root.update()  # Keep Tkinter responsive

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ---------------- Main ----------------
if __name__ == "__main__":
    setup_gui()
    root.mainloop()



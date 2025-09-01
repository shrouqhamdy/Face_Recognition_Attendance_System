# Face Recognition Attendance System

This project is a Python-based Face Recognition Attendance System designed for educational or organizational environments. It automates attendance tracking using facial recognition via a webcam, logs attendance in CSV files, manages a schedule of lectures, and can send attendance records to Discord through a webhook.

## Features

- **Real-time Face Recognition:** Detects and recognizes faces via camera feed.
- **Automated Attendance Logging:** Marks attendance and records it in a CSV file.
- **User-Friendly Interface:** Easy-to-use GUI for starting/stopping attendance, viewing logs, and notifications.
- **Attendance Notification:** Notifies users upon successful attendance registration.
- **User Registration:** Easily add new users by registering their face directly through the application interface.

## Quick Start

1. **Install Requirements**
   ```
   pip install -r requirements.txt
   ```
2. **Run the Application**
   ```
   python face_recognition_attendance.py
   ```
3. **Add Face Photos**
   - Place reference photos of users in the `photos/` directory.  
   - Or use the **registration feature** to capture and register new user faces from the app interface.


## Usage Screenshots

### Main Interface
![Main Window](https://github.com/shrouqhamdy/Face_Recognition_Attendance_System/blob/main/screenshots/main_window.jpg)

### Before and After Adding Lectures

| Before Recognition | After Recognition |
|--------------------|-------------------|
| ![Before](https://github.com/shrouqhamdy/Face_Recognition_Attendance_System/blob/main/screenshots/before.jpg) | ![After](https://github.com/shrouqhamdy/Face_Recognition_Attendance_System/blob/main/screenshots/after.jpg) |


### Attendance Notification
![Attendance Notification](https://github.com/shrouqhamdy/Face_Recognition_Attendance_System/blob/main/screenshots/attendance_notification.jpg)

### Attendance CSV File Example
![CSV File Sample](https://github.com/shrouqhamdy/Face_Recognition_Attendance_System/blob/main/screenshots/csv_file.jpg)

### User Registration Success
When a new student registers, such as Khalid, a confirmation message appears:

![Registration Success Example](https://github.com/shrouqhamdy/Face_Recognition_Attendance_System/blob/main/screenshots/registration_success.jpg)

## User Registration

- **Register New Users:**  
Use the registration button in the interface to add a new user. The system will prompt you to capture the user's face and input their name. The new user will be added to the attendance system automatically and their photo will be saved in the `photos` directory.

## File Overview

- `face_recognition_attendance.py`: Main script for running the system.
- `requirements.txt`: Python dependencies.
- `photos/`: Directory for users' face images.
- `screenshots/`: Sample UI and results.
- `shape_predictor_68_face_landmarks.dat`: Model file for facial landmarks detection.
- `short_success.mp3`: Audio notification for successful attendance.

## Notes

- The application uses dlib and OpenCV for face detection and recognition.
- Make sure your camera is properly connected for live recognition.

---

## Authors

- [shrouqhamdy](https://github.com/shrouqhamdy)  
- [khalidashraf630](https://github.com/khalidashraf630)

---

See [`face_recognition_attendance.py`](face_recognition_attendance.py) for details.

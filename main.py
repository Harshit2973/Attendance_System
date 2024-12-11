import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import csv

# Load images and create encodings
def load_images_from_folder(folder):
    images = []
    names = []
    if not os.path.exists(folder) or len(os.listdir(folder)) == 0:
        print("Error: No images found in the 'images' folder. Please add some images and try again.")
        exit()
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            face_locations = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if len(face_locations) == 0:
                print(f"Warning: No face detected in {filename}. Skipping this image.")
                continue
            images.append(img)
            names.append(os.path.splitext(filename)[0])
    return images, names

def encode_faces(images):
    encodings = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img_rgb)[0]
        encodings.append(enc)
    return encodings

# Log attendance
def mark_attendance(name):
    file_path = "attendance.csv"

    # Create file if it doesn't exist and add a header
    if not os.path.exists(file_path):
        with open(file_path, mode="w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Timestamp"])

    # Read and write attendance
    with open(file_path, mode="r+", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        lines = list(reader)
        recorded_names = [row[0] for row in lines[1:] if row]  # Skip the header

        if name not in recorded_names:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            writer = csv.writer(file)
            writer.writerow([name, timestamp])

# Display attendance log
def display_attendance():
    file_path = "attendance.csv"
    if not os.path.exists(file_path):
        print("No attendance recorded yet.")
        return

    print("\nAttendance Record:")
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            print(line.strip())

# Main Functionality
def main():
    path = "E:\Attendance_system\data"
    images, names = load_images_from_folder(path)
    if len(images) == 0:
        print("Error: No valid images with detectable faces found. Please check the 'images' folder.")
        exit()

    encodings = encode_faces(images)

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encodings, face_encoding)
            face_distances = face_recognition.face_distance(encodings, face_encoding)

            if any(matches):
                best_match_idx = np.argmin(face_distances)
                name = names[best_match_idx]

                mark_attendance(name)

                y1, x2, y2, x1 = face_location
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                y1, x2, y2, x1 = face_location
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("AI Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Display attendance log after exiting
    display_attendance()

if __name__ == "__main__":
    main()

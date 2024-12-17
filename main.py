import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
from PIL import Image
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import simpledialog, messagebox, ttk

# Directories and Files
DATA_DIR = "TrainingImage"
ATTENDANCE_DIR = "attendance"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Global face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def capture_image(roll_no, name):
    """Capture and save face images for training."""
    cap = cv2.VideoCapture(0)
    count = 0

    messagebox.showinfo("Instructions", "Look into the camera. Images will be captured automatically.")

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            file_path = os.path.join(DATA_DIR, f"{roll_no}_{name}_{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture Images", frame)

        if count >= 50:  # Capture more images for better training
            messagebox.showinfo("Success", "Image capture complete.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(subject, roll_no, name):
    """Mark attendance in the subject's CSV file."""
    subject_file = os.path.join(ATTENDANCE_DIR, f"{subject}.csv")
    
    # Create the file if it doesn't exist
    if not os.path.exists(subject_file):
        pd.DataFrame(columns=["Roll No", "Name", "Timestamp"]).to_csv(subject_file, index=False)

    # Read the existing attendance file
    df = pd.read_csv(subject_file)

    # Check if attendance is already marked for the current date
    today = datetime.now().date()
    if not ((df["Roll No"] == roll_no) & (pd.to_datetime(df["Timestamp"]).dt.date == today)).any():
        # Add a new entry to the attendance file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame([{"Roll No": roll_no, "Name": name, "Timestamp": timestamp}])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(subject_file, index=False)
        print(f"Attendance marked for {name} (Roll No: {roll_no}).")
    else:
        print(f"Attendance already marked for {name} (Roll No: {roll_no}).")

def train_faces():
    """Train the face recognizer using the saved face images."""
    faces, labels = [], []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".jpg"):
            try:
                path = os.path.join(DATA_DIR, file)
                img = Image.open(path).convert('L')  # Convert to grayscale
                img_resized = img.resize((100, 100))  # Consistent size
                img_np = np.array(img_resized, 'uint8')

                roll_no, _, _ = file.split("_")
                label = int(roll_no)
                faces.append(img_np)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if faces and labels:
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.write("trained_model.yml")  # Save the trained model
        messagebox.showinfo("Success", "Training complete.")
    else:
        messagebox.showerror("Error", "No valid faces found for training.")

import time  # Import time for the timer

def recognize_faces(subject):
    """Recognize faces and mark attendance, pause on recognition, and close after 20 seconds."""
    cap = cv2.VideoCapture(0)
    subject_file = os.path.join(ATTENDANCE_DIR, f"{subject}.csv")

    if not os.path.exists(subject_file):
        pd.DataFrame(columns=["Roll No", "Name", "Timestamp"]).to_csv(subject_file, index=False)

    # Load the trained model
    if not os.path.exists("trained_model.yml"):
        messagebox.showerror("Error", "No trained model found. Add and train students first.")
        return
    face_recognizer.read("trained_model.yml")

    recognized = False
    start_time = None  # To track when the face was recognized

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))

            try:
                label, confidence = face_recognizer.predict(face_resized)
                print(f"Detected Label: {label}, Confidence: {confidence:.2f}")  # Debug confidence

                if confidence < 70:  # Confidence threshold (lower is better)
                    # Use the label to get the student's name
                    name = student_names.get(str(label), "Unknown")  # Default to "Unknown" if label doesn't exist
                    roll_no = str(label)  # The label is actually the roll number
                    mark_attendance(subject, label, name)
                    color = (0, 255, 0)
                    
                    # Display the recognized face with Roll No and Name, excluding confidence
                    cv2.putText(frame, f"{roll_no} {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    if not recognized:  # Start timer on first recognition
                        recognized = True
                        start_time = time.time()

                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            except Exception as e:
                print(f"Recognition error: {e}")

        cv2.imshow("Recognize Faces", frame)

        # Check if face was recognized and 20 seconds have passed
        if recognized and time.time() - start_time > 20:
            print("20 seconds elapsed. Closing the camera.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow manual exit
            break

    cap.release()
    cv2.destroyAllWindows()


# GUI Functions
# A global dictionary to map label to student names
student_names = {}

def add_student_gui():
    add_student_window = tk.Toplevel(root)
    add_student_window.title("Add Student")
    
    # Roll Number input
    tk.Label(add_student_window, text="Enter Roll Number:", font=("Times", 12)).grid(row=0, column=0, padx=10, pady=10)
    roll_no_entry = tk.Entry(add_student_window, font=("Times", 12))
    roll_no_entry.grid(row=0, column=1, padx=10, pady=10)

    # Name input
    tk.Label(add_student_window, text="Enter Name:", font=("Times", 12)).grid(row=1, column=0, padx=10, pady=10)
    name_entry = tk.Entry(add_student_window, font=("Times", 12))
    name_entry.grid(row=1, column=1, padx=10, pady=10)

    def submit_student():
        roll_no = roll_no_entry.get()
        name = name_entry.get()

        if roll_no and name:  # Ensure both fields are filled
            capture_image(roll_no, name)  # Capture image with Roll No and Name
            student_names[roll_no] = name  # Store the name in the dictionary using roll_no as the key
            train_faces()  # Train the face recognition model
            add_student_window.destroy()  # Close the Add Student window
        else:
            messagebox.showerror("Error", "Please enter both Roll Number and Name.")

    # Submit button
    submit_button = tk.Button(add_student_window, text="Submit", font=("Times", 12), command=submit_student)
    submit_button.grid(row=2, columnspan=2, pady=20)


def mark_attendance_gui():
    subject = simpledialog.askstring("Input", "Enter Subject:")
    if subject:
        recognize_faces(subject)

def show_help():
    messagebox.showinfo("Help", "Instructions:\n1. Click 'Add Student' to capture images.\n2. Click 'Mark Attendance' to recognize faces.")

def resize_image(image_path, size):
    """Resize image to square size (width == height)."""
    image = Image.open(image_path)
    resized_image = image.resize((size, size))  # Resize to a square (size x size)
    return ImageTk.PhotoImage(resized_image)

def main():
    global root
    root = tk.Tk()
    root.title("Attendance Management System")
    # Set the background color of the window
    root.config(bg="grey")  # Change to desired color

    tk.Label(root, text=" Attendance Management System using Face Recognition", font=("times", 22) , bg="grey").grid(row=0, columnspan=2, pady=20) 

    # Resize images for Add Student and Mark Attendance buttons to square
    add_student_image = resize_image("Images/Add Student.png", 150)  # 150x150 square
    mark_attendance_image = resize_image("Images/Mark Attendance.png", 150)  # 150x150 square

    # Resize images for Help and Quit buttons to square
    help_image = resize_image("Images/Help.png", 150)  # 150x150 square
    quit_image = resize_image("Images/Quit.png", 150)  

    # Create buttons with resized square images
    button_size = 150  # Button size to match image size
    # Center the main window
    window_width = 1000
    window_height = 600
    root.geometry(f"{window_width}x{window_height}")
    root.resizable(False, False)  # Prevent window resizing

    # Style for buttons with RGB values
    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat", background="grey", foreground="black", font=("Times", 12))  # Button color (green) and text color (white)


    # Create Add Student Button
    add_student_button = ttk.Button(root, text="Add Student", image=add_student_image, compound="top", command=add_student_gui)
    add_student_button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    # Create Mark Attendance Button
    mark_attendance_button = ttk.Button(root, text="Mark Attendance", image=mark_attendance_image, compound="top", command=mark_attendance_gui)
    mark_attendance_button.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    # Create Help Button
    help_button = ttk.Button(root, text="Help", image=help_image, compound="top", command=show_help)
    help_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    # Create Quit Button
    quit_button = ttk.Button(root, text="Quit", image=quit_image, compound="top", command=root.quit)
    quit_button.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

    # Configure grid to make buttons expand equally and center them
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=0)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)

    # Keep a reference to the images to prevent garbage collection
    add_student_image_label = add_student_image
    mark_attendance_image_label = mark_attendance_image
    help_image_label = help_image
    quit_image_label = quit_image

    root.mainloop()

if __name__ == "__main__":
    main()
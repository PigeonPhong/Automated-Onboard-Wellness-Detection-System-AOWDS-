import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from deepface import DeepFace
import dlib
from scipy.spatial import distance
import time

# Initialize variables for emotions
anger, fear, disgust, surprise, happy, neutral = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
lambda_anger, lambda_fear, lambda_disgust, lambda_surprise, lambda_happy = 0.01, 0.01, 0.01, 0.01, 0.01
delta_t = 0.1
w1, w2, w3 = 0.3, 0.5, 0.2
flag1, flag2, flag3 = 1, 0, 1
beta = (w1 * flag1) + (w2 * flag2) + (w3 * flag3)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dlib's face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define thresholds for drowsiness detection
EAR_THRESHOLD = 0.15
MOR_THRESHOLD = 0.4
lambda_drowsiness = 0.01
frame_check = 20
flag = 0

# Function to calculate Eye Aspect Ratio
def calculate_EAR(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

# Function to calculate Mouth Open Ratio
def calculate_MOR(mouth):
    d = distance.euclidean(mouth[1], mouth[7])
    e = distance.euclidean(mouth[2], mouth[6])
    f = distance.euclidean(mouth[3], mouth[5])
    g = distance.euclidean(mouth[0], mouth[4])
    mor = (d + e + f) / (2.0 * g)
    return mor

# Function to update emotion level
def update_emotion(current_emotion, short_term_emotion, beta, delta_t, lambda_e):
    return current_emotion + beta * ((short_term_emotion - current_emotion) - lambda_e) * (1 - current_emotion) * delta_t * current_emotion

# Main application window
class CarPlayInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CarPlay Interface")
        self.setGeometry(100, 100, 1280, 720)


        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create a label for the webcam feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.video_label)

        # Start the question process immediately
        self.ask_question()

        # Initialize timer for webcam feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)  # Update every 30 milliseconds

    def ask_question(self):
        # Clear previous widgets
        for widget in self.findChildren(QWidget):
            widget.deleteLater()

        # Welcome label
        welcome_label = QLabel("Welcome, How are you today?", self)
        welcome_label.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.move(400, 50)

        # Questions
        self.questions = [
            "Are you in a rush?",
            "Are you stressed today?",
            "Are you tired?"
        ]
        self.current_question = 0

        # Display the current question
        self.question_label = QLabel(self.questions[self.current_question], self)
        self.question_label.setStyleSheet("font-size: 36px; font-weight: bold; color: white;")
        self.question_label.setAlignment(Qt.AlignCenter)
        self.question_label.move(400, 200)

        # Yes button
        self.yes_button = QPushButton("Yes", self)
        self.yes_button.setStyleSheet("background-color: #27ae60; color: white; font-size: 18px;")
        self.yes_button.move(400, 300)
        self.yes_button.clicked.connect(lambda: self.record_answer(1))

        # No button
        self.no_button = QPushButton("No", self)
        self.no_button.setStyleSheet("background-color: #c0392b; color: white; font-size: 18px;")
        self.no_button.move(600, 300)
        self.no_button.clicked.connect(lambda: self.record_answer(0))

    def record_answer(self, answer):
        self.answers[self.current_question] = answer
        self.current_question += 1

        if self.current_question < len(self.questions):
            self.ask_question()
        else:
            self.show_main_interface()

    def show_main_interface(self):
        # Clear previous widgets
        for widget in self.findChildren(QWidget):
            widget.deleteLater()

        # Main content layout
        self.setCentralWidget(QWidget())
        self.layout = QVBoxLayout()
        self.centralWidget().setLayout(self.layout)

        # Start updating the video feed
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB and display it in the QLabel
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    def closeEvent(self, event):
        # Release the webcam when the window is closed
        self.cap.release()
        event.accept()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarPlayInterface()
    window.show()
    sys.exit(app.exec_())
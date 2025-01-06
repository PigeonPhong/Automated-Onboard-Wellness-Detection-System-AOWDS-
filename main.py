import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
import dlib
from scipy.spatial import distance
import time
import threading
from moviepy import VideoFileClip

# Initialize variables for emotions, including neutral
anger, fear, disgust, surprise, happy, neutral = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Neutral remains static
lambda_anger, lambda_fear, lambda_disgust, lambda_surprise, lambda_happy = 0.01, 0.01, 0.01, 0.01, 0.01  # Decay rates
delta_t = 0.1  # Time interval
w1, w2, w3 = 0.3, 0.5, 0.2  # Weights for influence factors

# Pre-drive conditions flags for Rush, Tired, Stress
flag1, flag2, flag3 = 0, 1, 1

# Calculate beta (influence factor)
beta = (w1 * flag1) + (w2 * flag2) + (w3 * flag3)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dlib's face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#frame_check = 20  # Frame count for drowsiness alert
#flag = 0  # Flag for drowsiness alert

# Function to calculate Eye Aspect Ratio
def calculate_EAR(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

# Function to calculate Mouth Open Ratio
def calculate_MOR(mouth_landmarks):
    d = distance.euclidean(mouth_landmarks[1], mouth_landmarks[7])
    e = distance.euclidean(mouth_landmarks[2], mouth_landmarks[6])
    f = distance.euclidean(mouth_landmarks[3], mouth_landmarks[5])
    g = distance.euclidean(mouth_landmarks[0], mouth_landmarks[4])
    mor = (d + e + f) / (2.0 * g)
    return mor

# Function to update emotion level
def update_emotion(current_emotion, short_term_emotion, beta, delta_t, lambda_e):
    return current_emotion + beta * ((short_term_emotion - current_emotion) - lambda_e) * (1 - current_emotion) * delta_t * current_emotion

# Function to check drowsiness
def check_drowsiness(self, ear, mor, frame):
    """
    Check drowsiness level based on EAR and MOR values.
    
    Parameters:
        ear (float): Eye Aspect Ratio.
        mor (float): Mouth Open Ratio.
        frame (numpy.ndarray): Current video frame for displaying feedback.
    """
    # Constants
    MOR_THRESHOLD = 0.4  # Threshold for mouth open ratio
    YAWN_LIMIT = 5  # Number of yawns to increase EAR sensitivity
    EAR_SENSITIVITY_INCREMENT = 0.05  # Amount to reduce thresholds for sensitivity
    BASE_EAR_THRESHOLD = 0.15  # Base threshold for EAR
    BASE_EYE_AR_CONSEC_FRAMES = 20  # Base frame count for EAR detection

    # Persistent variables
    if not hasattr(self, 'yawn_counter'):
        self.yawn_counter = 0
    if not hasattr(self, 'EAR_THRESHOLD'):
        self.EAR_THRESHOLD = BASE_EAR_THRESHOLD
    if not hasattr(self, 'EYE_AR_CONSEC_FRAMES'):
        self.EYE_AR_CONSEC_FRAMES = BASE_EYE_AR_CONSEC_FRAMES
    if not hasattr(self, 'COUNTER'):
        self.COUNTER = 0

    # Check for yawning
    if mor > MOR_THRESHOLD:
        self.yawn_counter += 1
        cv2.putText(frame, "Yawning Detected!", (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Adjust EAR sensitivity if yawns exceed the limit
        if self.yawn_counter > YAWN_LIMIT:
            self.EAR_THRESHOLD = max(0.1, self.EAR_THRESHOLD - EAR_SENSITIVITY_INCREMENT)  # Prevent negative threshold
            self.EYE_AR_CONSEC_FRAMES = max(1, self.EYE_AR_CONSEC_FRAMES - 5)  # Ensure at least 1 frame

    # Check for drowsiness based on EAR
    if ear < self.EAR_THRESHOLD:
        self.COUNTER += 1
        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.play_video(video_key="drowsiness")  # Trigger action
    else:
        self.COUNTER = 0  # Reset the counter if EAR is above threshold

class CarPlayInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("CarPlay Interface")
        self.root.geometry("1280x720")  # 16:9 resolution
        self.root.configure(bg="#2c3e50")  # Darker blue background for a modern look


        # Initialize video paths
        self.video_paths = {
            "high_anger": "Ella/high_anger.mp4",
            "low_anger": "Ella/low_anger.mp4",
            "high_fear": "Ella/high_fear.mp4",
            "low_fear": "Ella/low_fear.mp4",
            "high_disgust": "Ella/high_disgust.mp4",
            "low_disgust": "Ella/low_disgust.mp4",
            "high_surprise": "Ella/high_surprise.mp4",
            "low_surprise": "Ella/low_surprise.mp4",
            "high_happy": "Ella/high_happy.mp4",
            "low_happy": "Ella/low_happy.mp4",
            "drowsiness": "Ella/drowsiness.mp4",
        }

        # Initialize video player
        self.video_capture = None
        self.video_label = tk.Label(self.root, bg="#2c3e50")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_label.pack_forget()  # Hide the video label initially

        # Initialize answers
        self.answers = [0, 0, 0]  # Store answers for the three questions
        self.current_question = 0  # Track the current question index
        self.show_camera = False  # Initialize show_camera attribute

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Timer variables
        self.start_time = time.time()  # Record the start time of the system
        self.timer_label = tk.Label(self.root, text="00:00:00", font=("Helvetica", 24), bg="#2c3e50", fg="#ecf0f1")
        self.timer_label.place(relx=0.95, rely=0.05, anchor="ne")  # Place the timer at the top-right corner
        self.update_timer()  # Start updating the timer

        # Start the question process immediately
        self.ask_question()

        self.last_alert_time = time.time()  # Initialize the last alert time

    def update_timer(self):
        # Calculate elapsed time
        elapsed_time = int(time.time() - self.start_time)
        # Format elapsed time as HH:MM:SS
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60
        timer_text = f"{hours:02}:{minutes:02}:{seconds:02}"
        # Update the timer label
        self.timer_label.config(text=timer_text)
        # Schedule the next update after 1 second
        self.root.after(1000, self.update_timer)

    def play_video(self, video_key=None, emotion_class=None, emotion_type=None):
        """
        Play a video based on the video key or emotion class and type.
        """
        if video_key is None:
            # Handle emotion videos
            video_key = f"{emotion_class}_{emotion_type}"
        
        # Debug: Print the video key
        print(f"Attempting to play video with key: {video_key}")

        # Get the video path
        video_path = self.video_paths.get(video_key)

        if not video_path:
            print(f"Video not found for key: {video_key}")
            return

        # Debug: Print the video path
        print(f"Playing video from path: {video_path}")

        # Function to play the video using MoviePy
        def play_video_thread():
            try:
                # Load the video
                clip = VideoFileClip(video_path)

                # Play the video
                clip.preview()  # This will open a window and play the video with sound

                # Close the video window when playback is complete
                clip.close()
            except Exception as e:
                print(f"Error playing video: {e}")

        # Start the video playback in a separate thread
        video_thread = threading.Thread(target=play_video_thread)
        video_thread.start()

    def update_video_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Convert the frame to RGB and display it in the Tkinter label
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            # Schedule the next frame update
            self.root.after(30, self.update_video_frame)
        else:
            # Video has ended, stop playback and return to the main ad frame
            self.video_capture.release()  # Release the video capture object
            self.video_label.pack_forget()  # Hide the video label
            self.ad_label.place(relx=0.5, rely=0.5, anchor="center")  # Show the ad frame

    def trigger_alert(self):
        # Determine the highest emotion
        emotions = {
            'anger': anger,
            'fear': fear,
            'disgust': disgust,
            'surprise': surprise,
            'happy': happy,
        }
        highest_emotion = max(emotions, key=emotions.get)
        highest_value = emotions[highest_emotion]

        # Classify the emotion as High or Low
        emotion_class = "high" if highest_value >= 0.5 else "low"

        # Play the corresponding video
        self.play_video(emotion_class=emotion_class, emotion_type=highest_emotion)


    def ask_question(self):
        # Clear previous question widgets
        for widget in self.root.winfo_children():
            widget.pack_forget()

        # Welcome label
        welcome_label = tk.Label(self.root, text="Welcome, How are you today?", 
                                  font=("Helvetica", 28, "bold"), bg="#2c3e50", fg="#ecf0f1")
        welcome_label.pack(pady=20)

        # Questions
        questions = [
            "Are you in a rush?",
            "Are you stressed today?",
            "Are you tired?"
        ]

        # Display the current question
        question_label = tk.Label(self.root, text=questions[self.current_question], 
                                  font=("Helvetica", 36, "bold"), bg="#2c3e50", fg="#ecf0f1")
        question_label.pack(pady=50)  # Increased padding for better spacing

        # Create a frame for buttons to center them
        button_frame = tk.Frame(self.root, bg="#2c3e50")
        button_frame.pack(pady=20)  # Add some vertical padding

        # Yes button
        yes_button = tk.Button(button_frame, text="Yes", command=lambda: self.record_answer(1), 
                               bg="#27ae60", fg="white", font=("Helvetica", 18), width=10, borderwidth=0)
        yes_button.pack(side=tk.LEFT, padx=20)  # Center the button in the frame
        yes_button.bind("<Enter>", lambda e: yes_button.config(bg="#2ecc71"))  # Hover effect
        yes_button.bind("<Leave>", lambda e: yes_button.config(bg="#27ae60"))  # Reset color

        # No button
        no_button = tk.Button(button_frame, text="No", command=lambda: self.record_answer(0), 
                              bg="#c0392b", fg="white", font=("Helvetica", 18), width=10, borderwidth=0)
        no_button.pack(side=tk.LEFT, padx=20)  # Center the button in the frame
        no_button.bind("<Enter>", lambda e: no_button.config(bg="#e74c3c"))  # Hover effect
        no_button.bind("<Leave>", lambda e: no_button.config(bg="#c0392b"))  # Reset color

    def record_answer(self, answer):
        self.answers[self.current_question] = answer  # Store the answer (0 or 1)
        
        # Update flags based on the current question
        if self.current_question == 0:
            global flag1
            flag1 = answer  # Set flag1 based on the answer to the first question
        elif self.current_question == 1:
            global flag2
            flag2 = answer  # Set flag2 based on the answer to the second question
        elif self.current_question == 2:
            global flag3
            flag3 = answer  # Set flag3 based on the answer to the third question

        self.current_question += 1  # Move to the next question

        if self.current_question < len(self.answers):
            # Ask the next question
            self.ask_question()
        else:
            # All questions answered, proceed to the main interface
            # Clear the question frame completely
            for widget in self.root.winfo_children():
                widget.destroy()
            self.show_main_interface()  # Show the main interface

    def show_main_interface(self):        
        # Main Content Frame
        self.buttons_frame = tk.Frame(self.root, bg="#001f3f")
        self.buttons_frame.pack(side=tk.LEFT, fill=tk.Y)  # Align buttons to the left

        self.camera_icon = ImageTk.PhotoImage(Image.open("icon/camera_icon.png").resize((50, 50)))  # Load camera icon
        
        # Camera toggle button with icon
        self.toggle_camera_btn = ttk.Button(self.buttons_frame, image=self.camera_icon, command=self.toggle_camera)
        self.toggle_camera_btn.image = self.camera_icon  # Keep a reference to avoid garbage collection
        self.toggle_camera_btn.grid(row=2, column=0, pady=10)

        # Main Content Frame
        self.content_frame = tk.Frame(self.root, bg="#001f3f")
        self.content_frame.pack(fill="both", expand=True)

        # Advertisement Screen
        self.ad_label = tk.Label(self.content_frame, text="Drive Safely!", font=("Arial", 36, "bold"), fg="white", bg="#001f3f")
        self.ad_label.place(relx=0.10, rely=0.10, anchor="center")

        # Timer label
        self.timer_label = tk.Label(self.content_frame, text="00:00:00", font=("Helvetica", 24), bg="#001f3f", fg="#ecf0f1")
        self.timer_label.pack(side=tk.TOP, pady=10)  # Place the timer at the top of the content frame

        # Reinitialize the video label
        self.video_label = tk.Label(self.content_frame, bg="#2c3e50")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_label.pack_forget()  # Hide the video label initially

        # Start updating the video feed
        self.update_video()

        # Restart the timer
        self.start_time = time.time()  # Reset the start time
        self.update_timer()  # Start updating the timer

    def create_button(self, icon, command, row):
        button = ttk.Button(self.buttons_frame, image=icon, command=command)
        button.image = icon  # Keep a reference to avoid garbage collection
        button.grid(row=row, column=0, padx=20, pady=10)


    def toggle_camera(self):
        if not self.show_camera:
            # Show camera feed
            self.video_label.pack()  # Show the video label
            self.ad_label.pack_forget()  # Hide the ad label
            self.show_camera = True
            self.toggle_camera_btn.config(text="Hide Camera")  # Optional: Change text if needed
        else:
            # Hide camera and show ad
            self.video_label.pack_forget()  # Hide the video label
            self.ad_label.pack()  # Show the ad label
            self.show_camera = False
            self.toggle_camera_btn.config(text="Show Camera")  # Optional: Change text if needed

    def update_video(self):
        global anger, fear, disgust, surprise, happy, neutral, flag
        ret, frame = self.cap.read()
        if ret or self.show_camera:
            # Initialize EAR and MOR with default values
            ear = 0.0  # Default EAR value
            mor = 0.0  # Default MOR value

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            # Process only the first detected face (if any)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_frame = frame[y:y+h, x:x+w]

                # Perform emotion analysis using DeepFace on the cropped face
                try:
                    result = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
                    # Normalize the short-term emotions to [0, 1] by dividing by 100
                    short_term_anger = result[0]['emotion']['angry'] / 100
                    short_term_fear = result[0]['emotion']['fear'] / 100
                    short_term_disgust = result[0]['emotion']['disgust'] / 100
                    short_term_surprise = result[0]['emotion']['surprise'] / 100
                    short_term_happy = result[0]['emotion']['happy'] / 100
                    short_term_neutral = result[0]['emotion']['neutral'] / 100

                    # Determine the dominant emotion
                    emotions = {
                        'anger': short_term_anger,
                        'fear': short_term_fear,
                        'disgust': short_term_disgust,
                        'surprise': short_term_surprise,
                        'happy': short_term_happy,
                        'neutral': short_term_neutral
                    }
                    dominant_emotion = max(emotions, key=emotions.get)

                except Exception as e:
                    print(f"Error in emotion analysis: {e}")
                    short_term_anger = short_term_fear = short_term_disgust = short_term_surprise = short_term_happy = short_term_neutral = 0
                    dominant_emotion = "No Emotion Detected"

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Display the dominant emotion in the rectangle
                cv2.putText(frame, f"Dominant: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Update emotion levels using the equations
                anger = update_emotion(anger, short_term_anger, beta, delta_t, lambda_anger)
                fear = update_emotion(fear, short_term_fear, beta, delta_t, lambda_fear)
                disgust = update_emotion(disgust, short_term_disgust, beta, delta_t, lambda_disgust)
                surprise = update_emotion(surprise, short_term_surprise, beta, delta_t, lambda_surprise)
                happy = update_emotion(happy, short_term_happy, beta, delta_t, lambda_happy)
                neutral = update_emotion(neutral, short_term_neutral, beta, delta_t, 0.5)

                # Drowsiness detection
                landmarks = face_landmark(gray, dlib.rectangle(x, y, x + w, y + h))

                # Initialize lists to store eye and mouth landmarks
                leftEye = []
                rightEye = []
                mouth = []

                # Draw lines around the left eye (landmarks 36 to 41)
                for n in range(36, 42):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    leftEye.append((x, y))
                    next_point = n + 1
                    if n == 41:
                        next_point = 36
                    x2 = landmarks.part(next_point).x
                    y2 = landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)  # Green lines for the left eye

                # Draw lines around the right eye (landmarks 42 to 47)
                for n in range(42, 48):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    rightEye.append((x, y))
                    next_point = n + 1
                    if n == 47:
                        next_point = 42
                    x2 = landmarks.part(next_point).x
                    y2 = landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)  # Green lines for the right eye

                # Draw lines around the mouth (landmarks 60 to 67)
                for n in range(60, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    mouth.append((x, y))
                    next_point = n + 1
                    if n == 67:
                        next_point = 60
                    x2 = landmarks.part(next_point).x
                    y2 = landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 0, 255), 1)  # Red lines for the mouth

                # Calculate Eye Aspect Ratio (EAR) for drowsiness detection
                leftEAR = calculate_EAR(leftEye)
                rightEAR = calculate_EAR(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Calculate Mouth Open Ratio (MOR) for drowsiness detection
                mor = calculate_MOR(mouth)

                # Check for drowsiness
                check_drowsiness(self,ear, mor, frame)

            # Display the emotion levels on the video feed at the bottom right corner
            height, width, _ = frame.shape  # Get the dimensions of the frame
            text_y = height - 30  # Start from the bottom of the frame

            # Display EAR and MOR values at the top-left corner
            cv2.putText(frame, f"EAR: {ear:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_y -= 30
            cv2.putText(frame, f"MOR: {mor:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_y -= 30
            
            cv2.putText(frame, f"Anger: {anger:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            text_y -= 30  # Move up for the next line
            cv2.putText(frame, f"Fear: {fear:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            text_y -= 30
            cv2.putText(frame, f"Disgust: {disgust:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_y -= 30
            cv2.putText(frame, f"Surprise: {surprise:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            text_y -= 30
            cv2.putText(frame, f"Happy: {happy:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            text_y -= 30
            cv2.putText(frame, f"Neutral: {neutral:.2f}", (width - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Display neutral level

            # Convert the frame to RGB for displaying in Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Check for alert every 3 minutes (180 seconds)
            current_time = time.time()
            if current_time - self.last_alert_time >= 60:  # 3 minutes(180)
                self.trigger_alert()  # Call the alert function
                self.last_alert_time = current_time  # Update the last alert time

        self.root.after(10, self.update_video)

    def close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarPlayInterface(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
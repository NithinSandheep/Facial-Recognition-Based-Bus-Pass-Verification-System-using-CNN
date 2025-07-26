import os  # Operating system interface for file/directory operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf  # Machine learning framework for neural networks
tf.get_logger().setLevel('ERROR')  # Suppress TF logger
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TF logging

import warnings  # System warnings handling
warnings.filterwarnings('ignore', category=UserWarning)

from flask import Flask, render_template, Response, request, jsonify, stream_with_context  # Web framework and utilities
import cv2  # OpenCV for image processing and computer vision
import dlib  # Face detection and recognition library
import os  # File and directory operations
import threading  # Multi-threading support
from datetime import datetime, timedelta  # Date and time operations
from apscheduler.schedulers.background import BackgroundScheduler  # Task scheduling
import subprocess  # Running external processes
import sys  # System-specific parameters and functions
import json  # JSON data handling
import time  # Time-related functions
import pickle  # Python object serialization
from tensorflow.keras.models import load_model  # Loading trained neural network models
import numpy as np  # Numerical operations and arrays
import random  # Random number generation

app = Flask(__name__, static_folder='static')

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
dataset_dir = r"dataset"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Global variables for capture status
capture_status = {
    "is_capturing": False,
    "count": 0,
    "total": 300,
    "current_person": None
}

camera = None
camera_lock = threading.Lock()

# Add a global variable for training progress
training_progress = {"epoch": 0, "total_epochs": 15}

# Load the model and label encoder at startup
face_model = load_model('facial_recognition_cnn.h5', compile=False)
face_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def get_camera():
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                # Try alternative backend
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return None
    return camera

def capture_images(student_id):  # Function to capture and save face images for registration
    global capture_status
    img_count = 0

    person_dir = os.path.join(dataset_dir, student_id)
    existing_images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    if existing_images:
        img_count = max(int(f.split('.')[0]) for f in existing_images)

    while capture_status["is_capturing"] and capture_status["count"] < capture_status["total"]:
        try:
            with camera_lock:
                ret, frame = get_camera().read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                
                for face in faces:
                    img_count += 1
                    capture_status["count"] += 1
                    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                    face_img = frame[y:y + h, x:x + w]
                    
                    face_filename = os.path.join(person_dir, f"{img_count}.jpg")
                    cv2.imwrite(face_filename, face_img)

                    if capture_status["count"] >= capture_status["total"]:
                        break
            
            threading.Event().wait(0.03)
        except Exception as e:
            print(f"Error in capture_images: {e}")
            continue

    capture_status["is_capturing"] = False

def generate_frames():  # Function to generate video frames for webcam feed
    while True:
        try:
            with camera_lock:
                ret, frame = get_camera().read()
                if not ret:
                    break
                
                if capture_status["is_capturing"]:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    for face in faces:
                        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            continue

@app.route('/')
def index():
    return render_template('bus_home.html')

@app.route('/register')
def register():
    return render_template('bus_register.html')

@app.route('/login')
def login():
    return render_template('bus_login.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_recognition_feed')
def face_recognition_feed():  # Function to handle real-time face recognition stream
    def generate():
        cap = cv2.VideoCapture(0)
        
        # Set exact dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 430)  # Reduced to account for message box
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                
                for face in faces:
                    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                    
                    # Add boundary checks
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if x + w > frame.shape[1]: w = frame.shape[1] - x
                    if y + h > frame.shape[0]: h = frame.shape[0] - y
                    
                    # Extract face with boundary checks
                    if w > 0 and h > 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Check if face_img is valid
                        if face_img is not None and face_img.size > 0:
                            try:
                                # Resize and preprocess
                                face_img = cv2.resize(face_img, (64, 64))
                                face_img = face_img / 255.0
                                face_img = np.expand_dims(face_img, axis=0)
                                
                                # Predict
                                predictions = face_model.predict(face_img)
                                predicted_idx = np.argmax(predictions[0])
                                confidence = predictions[0][predicted_idx] * 100
                                
                                # Rest of face recognition code...
                                if confidence < 70:
                                    ticket_message = "Unknown person"
                                    message_color = (0, 255, 255)
                                    box_color = (0, 255, 255)
                                else:
                                    name = le.inverse_transform([predicted_idx])[0]
                                    ticket_file = os.path.join(dataset_dir, name, "ticket_status.txt")
                                    ticket_status = 'no'
                                    expiry_date = None
                                    
                                    if os.path.exists(ticket_file):
                                        with open(ticket_file, 'r') as file:
                                            lines = file.readlines()
                                            for line in lines:
                                                if "Ticket Issued:" in line:
                                                    ticket_status = line.split(": ")[1].strip().lower()
                                                if "Expiry Date:" in line:
                                                    expiry_str = line.split(": ")[1].strip()
                                                    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S')

                                    if ticket_status == 'yes':
                                        if expiry_date:
                                            time_left = expiry_date - datetime.now()
                                            seconds_left = time_left.total_seconds()
                                            if seconds_left > 0:
                                                # Calculate days and format message
                                                days_left = int(seconds_left // 86400)
                                                if days_left > 0:
                                                    time_str = f"{days_left} day{'s' if days_left > 1 else ''} remaining"
                                                else:
                                                    time_str = "Less than a day left"
                                                
                                                ticket_message = f"{name}: Valid ({time_str})"
                                                message_color = (0, 255, 0)  # Green
                                                box_color = (0, 255, 0)
                                            else:
                                                ticket_message = f"{name}: No ticket"
                                                message_color = (0, 0, 255)  # Red
                                                box_color = (0, 0, 255)
                                        else:
                                            ticket_message = f"{name}: Has ticket"
                                            message_color = (0, 255, 0)
                                            box_color = (0, 255, 0)
                                    else:
                                        ticket_message = f"{name}: No ticket"
                                        message_color = (0, 0, 255)
                                        box_color = (0, 0, 255)

                                # Draw rectangle and message
                                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                                message_height = 50
                                cv2.rectangle(frame, (0, frame.shape[0] - message_height), 
                                            (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                                cv2.putText(frame, ticket_message, (10, frame.shape[0] - 10), 
                                          cv2.FONT_ITALIC, 1, message_color, 2)
                            except Exception as e:
                                print(f"Error processing face: {e}")
                                continue

                # Convert to jpg for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate: {e}")
        finally:
            cap.release()
            
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():  # Function to handle the image capture process during registration
    global capture_status
    if capture_status["is_capturing"]:
        return jsonify({"status": "error", "message": "Capture already in progress"})

    data = request.json
    username = data.get('username')
    college_id = data.get('collegeId')
    password = data.get('password')
    
    person_dir = os.path.join(dataset_dir, username)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        
    # Save credentials with college ID
    credentials_file = os.path.join(dataset_dir, "credentials.txt")
    with open(credentials_file, "a") as file:
        file.write(f"Person ID: {username}\n")
        file.write(f"College ID: {college_id}\n")
        file.write(f"Password: {password}\n---\n")
        
    # Set default ticket status with college ID
    with open(os.path.join(person_dir, "ticket_status.txt"), "w") as file:
        file.write(f"Person ID: {username}\n")
        file.write(f"College ID: {college_id}\n")
        file.write("Ticket Issued: no\n")
    
    capture_status = {
        "is_capturing": True,
        "count": 0,
        "total": 300,
        "current_person": username
    }
    
    thread = threading.Thread(target=capture_images, args=(username,))
    thread.daemon = True
    thread.start()
        
    return jsonify({"status": "success"})

@app.route('/capture_status')
def get_capture_status():  # Function to get current status of image capture process
    percentage = (capture_status["count"] / capture_status["total"]) * 100
    rounded_percentage = int(percentage // 10 * 10)
    
    return jsonify({
        "is_capturing": capture_status["is_capturing"],
        "count": rounded_percentage,
        "total": 100
    })

@app.route('/verify_login', methods=['POST'])
def verify_login():  # Function to verify user login credentials
    data = request.json
    student_id = data.get('studentId')
    password = data.get('password')
    
    try:
        # Get user details from credentials file
        college_id = ''
        credentials_file = os.path.join(dataset_dir, "credentials.txt")
        if os.path.exists(credentials_file):
            with open(credentials_file, 'r') as file:
                content = file.read()
                entries = content.split('---\n')
                for entry in entries:
                    if f"Person ID: {student_id}" in entry:
                        # Check password first
                        if f"Password: {password}" in entry:
                            # If password matches, look for College ID
                            for line in entry.split('\n'):
                                if "College ID:" in line:
                                    college_id = line.split(':', 1)[1].strip()
                                    return jsonify({
                                        "status": "success",
                                        "name": student_id,
                                        "collegeId": college_id
                                    })
                            # If no College ID found but credentials match
                            return jsonify({
                                "status": "success",
                                "name": student_id,
                                "collegeId": student_id  # Use Person ID as fallback
                            })
    except Exception as e:
        print(f"Error in verify_login: {e}")
        
    return jsonify({
        "status": "error",
        "message": "Invalid credentials"
    })

@app.route('/verify', methods=['POST'])
def verify():  # Function to verify user credentials during authentication
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    try:
        # First check credentials
        credentials_file = os.path.join(dataset_dir, "credentials.txt")
        if not os.path.exists(credentials_file):
            return jsonify({
                "status": "error",
                "message": "No registered users found"
            })
            
        with open(credentials_file, 'r') as file:
            content = file.read()
            entries = content.split('---\n')
            
            for entry in entries:
                if not entry.strip():
                    continue
                lines = entry.strip().split('\n')
                entry_data = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        entry_data[key.strip()] = value.strip()
                
                if (entry_data.get('Person ID') == username and 
                    entry_data.get('Password') == password):
                    
                    # Check if ticket is expired
                    check_ticket_expiry()  # Run expiry check
                    
                    # Return ticket status
                    ticket_file = os.path.join(dataset_dir, username, "ticket_status.txt")
                    if os.path.exists(ticket_file):
                        with open(ticket_file, 'r') as tf:
                            if 'Expired' in tf.read():
                                return jsonify({
                                    "status": "warning",
                                    "message": "Your ticket has expired. Please renew."
                                })
                    
                    return jsonify({
                        "status": "success",
                        "message": "Valid credentials"
                    })
                    
        return jsonify({
            "status": "error",
            "message": "Invalid credentials"
        })
    except Exception as e:
        print(f"Error in verify: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/book_ticket', methods=['POST'])
def book_ticket():  # Function to process ticket booking requests
    try:
        data = request.json
        username = data.get('username')
        college_id = data.get('collegeId')  # Get college ID from request
        location = data.get('location')
        price = data.get('price')
        isReturn = data.get('isReturn', False)
        duration = 259200 if isReturn else 86400  # 3 days or 1 day in seconds
        
        user_dir = os.path.join(dataset_dir, username)
        if not os.path.exists(user_dir):
            return jsonify({"status": "error", "message": "User not found"})

        current_time = datetime.now()
        expiry_time = current_time + timedelta(seconds=duration)

        # Update booking_details.txt
        booking_file = os.path.join(user_dir, "booking_details.txt")
        with open(booking_file, 'w') as file:
            file.write(f"Location: {location}\n")
            file.write(f"Price: {price}\n")
            file.write(f"Issue Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Expiry Date: {expiry_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Ticket Type: {'Return' if isReturn else 'Single'}\n")
            file.write(f"Duration: {duration}\n")

        # Update history.txt
        history_file = os.path.join(user_dir, "history.txt")
        with open(history_file, 'a') as file:
            file.write("---\n")
            file.write(f"Location: {location}\n")
            file.write(f"Price: {price}\n")
            file.write(f"Issue Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Expiry Date: {expiry_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Ticket Type: {'Return' if isReturn else 'Single'}\n")
            file.write(f"Duration: {duration}\n")

        # Update ticket_status.txt
        status_file = os.path.join(user_dir, "ticket_status.txt")
        with open(status_file, 'w') as file:
            file.write(f"Person ID: {username}\n")
            file.write("Ticket Issued: yes\n")
            file.write(f"Expiry Date: {expiry_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Ticket Type: {'Return' if isReturn else 'Single'}\n")
            file.write(f"Duration: {duration}\n")

        return jsonify({
            "status": "success",
            "redirect": f"/booking_success?username={username}&location={location}&price={price}&duration={duration}&isReturn={str(isReturn).lower()}&collegeId={college_id}"
        })
    except Exception as e:
        print(f"Error booking ticket: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/update_ticket_status', methods=['POST'])
def update_ticket_status():  # Function to update ticket status for a user
    data = request.json
    username = data.get('username')
    
    try:
        # Direct path to user's ticket status file
        person_dir = os.path.join(dataset_dir, username)
        ticket_file = os.path.join(person_dir, "ticket_status.txt")
        
        if os.path.exists(person_dir):
            with open(ticket_file, "w") as tf:
                tf.write(f"Person ID: {username}\nTicket Issued: yes\n")
            return jsonify({"status": "success"})
        
        return jsonify({
            "status": "error",
            "message": "User directory not found"
        })
    except Exception as e:
        print(f"Error in update_ticket_status: {e}")  # Debug print
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/booking_success')
def booking_success():  # Function to handle successful ticket booking and display receipt
    username = request.args.get('username')
    location = request.args.get('location')
    price = request.args.get('price')
    isReturn = request.args.get('isReturn') == 'true'
    duration = 259200 if isReturn else 86400
    
    # Get user details from credentials file
    college_id = ''
    try:
        credentials_file = os.path.join(dataset_dir, "credentials.txt")
        if os.path.exists(credentials_file):
            with open(credentials_file, 'r') as file:
                content = file.read()
                entries = content.split('---\n')
                for entry in entries:
                    if f"Person ID: {username}" in entry:
                        for line in entry.split('\n'):
                            if "College ID:" in line:
                                college_id = line.split(':', 1)[1].strip()
                                break
    except Exception as e:
        print(f"Error getting college ID: {e}")
    
    booking_date = datetime.now().strftime("%B %d, %Y")
    booking_time = datetime.now().strftime("%I:%M %p")
    expiry_datetime = datetime.now() + timedelta(seconds=duration)
    expiry_date = expiry_datetime.strftime("%B %d, %Y %I:%M %p")
    
    return render_template('receipt.html',
                         username=username,
                         name=username,
                         collegeId=college_id,
                         location=location,
                         price=price,
                         isReturn=isReturn,
                         booking_date=booking_date,
                         booking_time=booking_time,
                         expiry_date=expiry_date)

@app.route('/shutdown')
def shutdown():  # Function to properly close camera and cleanup resources
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
        cv2.destroyAllWindows()  # Add this line to close any OpenCV windows
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in shutdown: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():  # Function to stop ongoing image capture process
    global capture_status
    try:
        capture_status["is_capturing"] = False
        capture_status["count"] = 0
        return jsonify({
            "status": "success",
            "message": "Capture stopped"
        })
    except Exception as e:
        print(f"Error in stop_capture: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Modify the check_ticket_expiry function to run more frequently
def check_ticket_expiry():  # Function to check and update expired tickets
    try:
        for person_id in os.listdir(dataset_dir):
            person_dir = os.path.join(dataset_dir, person_id)
            if os.path.isdir(person_dir):
                ticket_file = os.path.join(person_dir, "ticket_status.txt")
                if os.path.exists(ticket_file):
                    with open(ticket_file, 'r') as file:
                        lines = file.readlines()
                        expiry_date = None
                        for line in lines:
                            if "Expiry Date:" in line:
                                expiry_str = line.split(": ")[1].strip()
                                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S')
                                
                        if expiry_date and datetime.now() > expiry_date:
                            # Ticket has expired, reset to default status
                            with open(ticket_file, 'w') as file:
                                file.write(f"Person ID: {person_id}\n")
                                file.write("Ticket Issued: no\n")
    except Exception as e:
        print(f"Error checking ticket expiry: {e}")



@app.route('/ticket_history/<username>')
def ticket_history(username):  # Function to retrieve user's ticket booking history
    user_dir = os.path.join(dataset_dir, username)
    history_file = os.path.join(user_dir, "history.txt")
    tickets = []
    
    if (os.path.exists(history_file)):
        with open(history_file, 'r') as file:
            content = file.read()
            ticket_entries = content.split('---\n')
            
            for entry in ticket_entries:
                if entry.strip():
                    ticket_data = {}
                    lines = entry.strip().split('\n')
                    for line in lines:
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            ticket_data[key.lower()] = value.strip()
                    
                    # Handle empty or missing expiry date
                    expiry_date_str = ticket_data.get('expiry date')
                    if expiry_date_str:
                        try:
                            expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d %H:%M:%S')
                            is_active = expiry_date > datetime.now()
                        except ValueError:
                            expiry_date = datetime.now()
                            is_active = False
                    else:
                        expiry_date = datetime.now()
                        is_active = False
                    
                    # Clean up price value by removing any existing "Rs." prefix
                    price_value = ticket_data.get('price', '0')
                    price_value = price_value.replace('Rs.', '').replace('₹', '').strip()
                    
                    tickets.append({
                        'route': ticket_data.get('location', ''),
                        'price': price_value,
                        'purchaseDate': ticket_data.get('issue date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                        'expiryDate': expiry_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': ticket_data.get('duration', ''),
                        'reference': '%010x' % random.randrange(16**10),
                        'isActive': is_active
                    })
            
            # Sort tickets: active first, then by purchase date
            tickets.sort(key=lambda x: (
                not x['isActive'],  # Active tickets first
                -datetime.strptime(x['purchaseDate'], '%Y-%m-%d %H:%M:%S').timestamp()  # Most recent second
            ))
            
            # Keep only the most recent 5 tickets
            tickets = tickets[:5]
            
            # Update history.txt with only the kept tickets
            with open(history_file, 'w') as file:
                for ticket in tickets:  # No need to reverse since we want newest first
                    file.write("---\n")
                    file.write(f"Location: {ticket['route']}\n")
                    file.write(f"Price: Rs.{ticket['price']}\n")
                    file.write(f"Issue Date: {ticket['purchaseDate']}\n")
                    file.write(f"Expiry Date: {ticket['expiryDate']}\n")
                    file.write(f"Duration: {ticket['duration']}\n")
    
    return jsonify({'tickets': tickets})

@app.route('/train_progress')
def train_progress():  # Function to track and report model training progress
    def generate():
        while True:
            # Send current progress
            data = {
                "progress": True,
                "epoch": training_progress["epoch"],
                "total_epochs": training_progress["total_epochs"]
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            # Check if training is complete
            if training_progress["epoch"] >= training_progress["total_epochs"]:
                data = {"complete": True, "success": True}
                yield f"data: {json.dumps(data)}\n\n"
                break
                
            time.sleep(1)  # Check progress every second
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/train_model', methods=['POST'])
def train_model():  # Function to initiate and manage model training process
    try:
        global training_progress, face_model, le  # Add model and encoder to global
        training_progress = {"epoch": 0, "total_epochs": 15}
        
        python_executable = sys.executable
        print("Starting model training...")
        
        process = subprocess.Popen([python_executable, 'trainer.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True,
                                 encoding='utf-8',
                                 errors='replace')
        
        # Monitor output and process completion
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
                if "Epoch" in output:
                    try:
                        epoch = int(output.split("Epoch")[1].split("/")[0])
                        training_progress["epoch"] = epoch
                    except:
                        pass
            if process.poll() is not None:
                break
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            # Reload the model and encoder after successful training
            face_model = load_model('facial_recognition_cnn.h5')
            with open('label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            training_progress["epoch"] = training_progress["total_epochs"]
            return jsonify({"status": "success"})
        else:
            return jsonify({
                "status": "error",
                "message": f"Training failed: {stderr}"
            })
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/expire_ticket', methods=['POST'])
def expire_ticket():  # Function to manually expire a ticket
    try:
        data = request.json
        username = data.get('username')
        purchase_date = data.get('purchaseDate')
        
        user_dir = os.path.join(dataset_dir, username)
        
        # Update ticket_status.txt
        status_file = os.path.join(user_dir, "ticket_status.txt")
        if os.path.exists(status_file):
            with open(status_file, 'w') as file:
                file.write(f"Person ID: {username}\n")
                file.write("Ticket Issued: no\n")
        
        # Update history.txt to mark ticket as expired
        history_file = os.path.join(user_dir, "history.txt")
        if os.path.exists(history_file):
            with open(history_file, 'r') as file:
                content = file.read()
                entries = content.split('---\n')
                updated_content = []
                
                for entry in entries:
                    if entry.strip():
                        if f"Issue Date: {purchase_date}" in entry:
                            # Update expiry date to current time
                            updated_entry = []
                            for line in entry.split('\n'):
                                if "Expiry Date:" in line:
                                    updated_entry.append(f"Expiry Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                else:
                                    updated_entry.append(line)
                            updated_content.append('\n'.join(updated_entry))
                        else:
                            updated_content.append(entry)
                            
            # Write back updated content
            with open(history_file, 'w') as file:
                file.write('---\n'.join(updated_content))
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Update the scheduler to run every 10 seconds instead of 24 hours
scheduler = BackgroundScheduler()
scheduler.add_job(func=check_ticket_expiry, trigger="interval", seconds=10)  # Changed from hours=24
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True)
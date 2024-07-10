# Import necessary libraries
from flask import Flask, request, jsonify  # Flask is a web framework for Python
from flask_cors import CORS  # CORS is a Flask extension for handling Cross-Origin Resource Sharing
import cv2  # OpenCV for computer vision tasks
import numpy as np  # NumPy for numerical operations
import mediapipe as mp  # MediaPipe for pose detection and other tasks
from mediapipe.tasks import python  # MediaPipe tasks
from mediapipe.tasks.python import vision  # MediaPipe vision tasks
from mediapipe.framework.formats import landmark_pb2  # MediaPipe landmark formats

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

# Function to process video and detect poses
@app.route('/process', methods=['POST'])
def video_processer():
    # Extract data from the JSON payload
    data = request.get_json()
    users_side = data.get('users_side')
    video_url = data.get('video_url')

    LR = 0
    if users_side == "Left":
        users_side = 0
    else:
        users_side = 1

    # Initialize variables for tracking arm and leg movements
    after_y = 1
    after_ya = 1
    pre_1 = 0
    pre_2 = 0
    pre_1a = 0
    pre_2a = 0
    feedback = []  # List to store feedback messages
    
    # Set up MediaPipe drawing utility and model options
    mpDraw = mp.solutions.drawing_utils
    model_path = "pose_landmarker_full.task"
    num_poses = 2
    min_pose_detection_confidence = 0.5
    min_pose_presence_confidence = 0.5
    min_tracking_confidence = 0.5

    # Configure MediaPipe pose landmarker options
    base_options = python.BaseOptions(model_asset_path=model_path) 
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    
    # Create pose landmarker from options
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    # Open video URL
    cap = cv2.VideoCapture(video_url)
    
    # Get video properties
    fps = int(cap.get(5))  # Frames per second
    frame_width = int(cap.get(3))  # Frame width
    frame_height = int(cap.get(4))  # Frame height
    frame_size = (frame_width, frame_height)
    
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if ret:
            # Convert frame to RGB and create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)  # Get timestamp
            result = landmarker.detect_for_video(mp_image, timestamp_ms)  # Detect poses
            pose_landmarks_list = result.pose_landmarks  # List of detected poses
            frame = np.copy(mp_image.numpy_view())  # Copy image data
            noses = [pose_landmarks_list[person][0].x for person in range(len(pose_landmarks_list))]  # Get nose positions
            noses.sort()  # Sort nose positions
            
            # Process each detected pose
            for person in range(len(pose_landmarks_list)):
                pose_landmarks = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z) for landmark in pose_landmarks_list[person]
                ])
                if len(noses) <= 1:  # Skip if there are less than two poses
                    continue
                
                if pose_landmarks_list[person][0].x == noses[LR]:  # If current pose matches user's side
                    # Get hip, knee, and ankle landmarks
                    hip = pose_landmarks_list[person][23]
                    ankle = pose_landmarks_list[person][27]
                    knee = pose_landmarks_list[person][25]
                    angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))  # Calculate angle
                    if angle > 90:
                        feedback.append("You did a lunge")  # Provide feedback
                    
                    arm_y = pose_landmarks_list[person][16].y  # Get arm y-coordinate
                    if arm_y - after_y > 0.1:
                        feedback.append("You did a parry")  # Provide feedback
                    after_y = arm_y
                    
                    knee_1 = pose_landmarks_list[person][26].x  # Get knee x-coordinates
                    knee_2 = pose_landmarks_list[person][25].x
                    if (knee_1 < knee_2 and pre_1 > pre_2) or (knee_1 > knee_2 and pre_1 < pre_2):
                        feedback.append("You did a fleche")  # Provide feedback
                    pre_1 = knee_1
                    pre_2 = knee_2
                else:  # If current pose matches opponent's side
                    hipa = pose_landmarks_list[person][23]
                    anklea = pose_landmarks_list[person][27]
                    kneea = pose_landmarks_list[person][25]
                    anglea = calculate_angle((hipa.x, hipa.y), (kneea.x, kneea.y), (anklea.x, anklea.y))
                    if anglea > 90:
                        feedback.append("Your opponent lunged")  # Provide feedback
                    
                    arm_ya = pose_landmarks_list[person][16].y  # Get arm y-coordinate
                    if arm_ya - after_ya > 0.1:
                        feedback.append("Your opponent parried")  # Provide feedback
                    after_ya = arm_ya
                    
                    knee_1a = pose_landmarks_list[person][26].x  # Get knee x-coordinates
                    knee_2a = pose_landmarks_list[person][25].x
                    if (knee_1a < knee_2a and pre_1a > pre_2a) or (knee_1a > knee_2a and pre_1a < pre_2a):
                        feedback.append("Your opponent did a fleche")  # Provide feedback
                    pre_1a = knee_1a
                    pre_2a = knee_2a
        else:
            print("Stream disconnected")  # Print message if stream is disconnected
            break
    
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Destroy all OpenCV windows

    if len(feedback) == 0:
        feedback = ["Could not detect any moves in this footage."]
    
    print(feedback)

    return jsonify(feedback)  # Return feedback list

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0')

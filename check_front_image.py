import torch
from PIL import Image, ImageEnhance, ImageDraw
import cv2
import numpy as np
from preprocess.openpose.run_openpose import OpenPose
from rembg import remove
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

device = "cuda" 
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(device)

def enhance_image(image):
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply histogram equalization to improve contrast
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Apply slight Gaussian blur to reduce noise
    img_enhanced = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    
    # Increase sharpness
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_enhanced = cv2.filter2D(img_enhanced, -1, kernel)
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB))

def load_and_preprocess_image(image_path):
    # Open the image
    human_img = Image.open(image_path)
    
    # Remove background
    human_img = remove(human_img)
    
    #Convert to RGB if the image is in RGBA mode
    if human_img.mode == 'RGBA':
        background = Image.new('RGBA', human_img.size, (255, 255, 255))
        human_img = Image.alpha_composite(background, human_img).convert('RGB')
    
    # Resize the image
    human_img = human_img.resize((384, 512))
    
    # Enhance the image
    # human_img = enhance_image(human_img)

    # Increae the saturation and colours in the image
    enhancer = ImageEnhance.Color(human_img)
    human_img = enhancer.enhance(1.5)
    
    return human_img

def detect_faces(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_keypoints_and_faces(openpose_model, image):
    keypoints = openpose_model(image)
    faces = detect_faces(image)
    
    pose_keypoints = keypoints['pose_keypoints_2d']
    
    # Map faces to keypoints
    face_mapping = {}
    for i, (x, y, w, h) in enumerate(faces):
        face_center = (x + w//2, y + h//2)
        closest_keypoint_index = np.argmin([
            ((kp[0] - face_center[0])**2 + (kp[1] - face_center[1])**2)
            if kp[0] != 0 and kp[1] != 0 else float('inf')
            for kp in pose_keypoints[:18]  # Consider only body keypoints
        ])
        face_mapping[f'face_{i+1}'] = {
            'bbox': (x, y, w, h),
            'center': face_center,
            'closest_keypoint_index': closest_keypoint_index
        }
    
    return pose_keypoints, face_mapping

def analyze_pose(pose_keypoints, face_mapping, image_shape):
    keypoint_dict = {}
    keypoint_names = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear"
    ]

    for i, keypoint in enumerate(pose_keypoints):
        if keypoint[0] != 0 and keypoint[1] != 0:
            keypoint_dict[keypoint_names[i]] = (int(keypoint[0]), int(keypoint[1]))

    analysis = {
        'is_single_human': False,
        'no_of_humans_detected': 0,
        'is_standing_straight': False,
        'is_facing_front': False,
        'is_hands_straight_or_slightly_bent': False,
        'is_hands_in_front_of_torso': False,
        'torso_detected': False
    }

    # Check if torso is detected
    torso_keypoints = ["neck", "right_shoulder", "left_shoulder", "right_hip", "left_hip"]
    analysis['torso_detected'] = all(kp in keypoint_dict for kp in torso_keypoints)

    # Check if standing straight
    if all(kp in keypoint_dict for kp in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        shoulder_slope = abs((keypoint_dict['left_shoulder'][1] - keypoint_dict['right_shoulder'][1]) / 
                             (keypoint_dict['left_shoulder'][0] - keypoint_dict['right_shoulder'][0] + 1e-6))
        hip_slope = abs((keypoint_dict['left_hip'][1] - keypoint_dict['right_hip'][1]) / 
                        (keypoint_dict['left_hip'][0] - keypoint_dict['right_hip'][0] + 1e-6))
        analysis['is_standing_straight'] = shoulder_slope < 0.15 and hip_slope < 0.15

    # Check if facing front
    if all(kp in keypoint_dict for kp in ['left_shoulder', 'right_shoulder']):
        shoulder_width = abs(keypoint_dict['left_shoulder'][0] - keypoint_dict['right_shoulder'][0])
        analysis['is_facing_front'] = shoulder_width > 0.15 * image_shape[1]

    # Check if hands are in front of torso
    if analysis['torso_detected'] and 'left_wrist' in keypoint_dict and 'right_wrist' in keypoint_dict:
        # Define the torso area as a polygon
        torso_polygon = Polygon([
            keypoint_dict['right_shoulder'],
            keypoint_dict['left_shoulder'],
            keypoint_dict['left_hip'],
            keypoint_dict['right_hip']
        ])

        # Check if either hand is within the torso polygon
        left_hand = Point(keypoint_dict['left_wrist'])
        right_hand = Point(keypoint_dict['right_wrist'])

        analysis['is_hands_in_front_of_torso'] = torso_polygon.contains(left_hand) or torso_polygon.contains(right_hand)

    # Check if hands are straight or slightly bent
    if all(kp in keypoint_dict for kp in ['left_shoulder', 'left_elbow', 'left_wrist', 
                                          'right_shoulder', 'right_elbow', 'right_wrist']):
        def calculate_arm_angle(shoulder, elbow, wrist):
            upper_arm = np.array(elbow) - np.array(shoulder)
            forearm = np.array(wrist) - np.array(elbow)
            angle = np.degrees(np.arccos(np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))))
            return angle

        left_angle = calculate_arm_angle(keypoint_dict['left_shoulder'], keypoint_dict['left_elbow'], keypoint_dict['left_wrist'])
        right_angle = calculate_arm_angle(keypoint_dict['right_shoulder'], keypoint_dict['right_elbow'], keypoint_dict['right_wrist'])

        # Consider arms straight or slightly bent if the angle is between 160 and 200 degrees
        analysis['is_hands_straight_or_slightly_bent'] = (160 <= left_angle <= 200) or (160 <= right_angle <= 200)

    return analysis

def detect_humans(image):
    """Detect humans in the image using YOLOv5"""
    model = YOLO('yolov5su.pt')
    img_np = np.array(image)
    results = model(img_np)
    detections = results[0].boxes.data
    person_detections = detections[detections[:, 5] == 0]
    return person_detections

def visualize_results(image, pose_keypoints, face_mapping, person_detections):
    img_np = np.array(image)
    img_with_keypoints = img_np.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    # Draw pose keypoints
    for i, keypoint in enumerate(pose_keypoints):
        if keypoint[0] != 0 and keypoint[1] != 0:
            x, y = map(int, keypoint)
            cv2.circle(img_with_keypoints, (x, y), 5, colors[i % len(colors)], -1)

    # Draw face bounding boxes and labels
    for face_id, face_info in face_mapping.items():
        x, y, w, h = face_info['bbox']
        cv2.rectangle(img_with_keypoints, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_with_keypoints, face_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Draw bounding boxes around detected persons
    for det in person_detections:
        x1, y1, x2, y2, conf, _ = det
        cv2.rectangle(img_with_keypoints, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_with_keypoints, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return Image.fromarray(img_with_keypoints)

def visualize_step(image, title):
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), title, fill=(255, 0, 0))
    image.show(title)

def analyze_image(image_path):

    # Load and preprocess image
    human_img = load_and_preprocess_image(image_path)
    # visualize_step(human_img, "Preprocessed Image")

    # Detect humans using YOLOv5
    person_detections = detect_humans(human_img)
    no_of_humans_detected = len(person_detections)

    # Visualize YOLOv5 detections
    img_with_detections = human_img.copy()
    draw = ImageDraw.Draw(img_with_detections)
    for det in person_detections:
        x1, y1, x2, y2 = map(int, det[:4])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    # visualize_step(img_with_detections, f"YOLOv5 Detections: {no_of_humans_detected} humans")

    # Detect keypoints and faces
    pose_keypoints, face_mapping = detect_keypoints_and_faces(openpose_model, human_img)

    # Visualize keypoints and faces
    img_with_keypoints = visualize_results(human_img, pose_keypoints, face_mapping, person_detections)
    # visualize_step(img_with_keypoints, "Pose Keypoints and Face Detection")

    # Analyze pose
    analysis = analyze_pose(pose_keypoints, face_mapping, human_img.size)

    # Update analysis based on YOLOv5 detection
    analysis['no_of_humans_detected'] = no_of_humans_detected
    analysis['is_single_human'] = no_of_humans_detected == 1

    return analysis

# Main execution
# if __name__ == "__main__":

#     # Run a loop on all images in FrontImages folder and store the results in a csv file
#     image_path = "FrontImages/image12.jpeg"
    
#     assessment = analyze_image(image_path)
    
#     if assessment is not None:
#         print("\nPosture Assessment:")
#         for key, value in assessment.items():
#             if isinstance(value, bool):
#                 print(f"{key}: {'Yes' if value else 'No'}")
#             else:
#                 print(f"{key}: {value}")
#     else:
#         print("Failed to process the image.")

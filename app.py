# import streamlit as st
# import mediapipe as mp
# import cv2
# import numpy as np
# import tempfile
# import os

# # 1. Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# def calculate_angle(a, b, c):
#     """Calculates the angle between three points (a is the vertex)"""
#     a = np.array(a) # First point (e.g., Shoulder)
#     b = np.array(b) # Mid point (e.g., Hip)
#     c = np.array(c) # End point (e.g., Knee)
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)
    
#     if angle > 180.0:
#         angle = 360-angle
#     return angle

# # 2. Streamlit UI Setup
# st.set_page_config(page_title="Golf AI Pro", page_icon="⛳")
# st.title("⛳ AI Golf Swing Analyzer")
# st.markdown("""
# **1-Hour Sprint Project**: Upload a face-on swing video. 
# The AI will map your joints and calculate your lead-arm/shoulder rotation.
# """)

# # Sidebar instructions
# with st.sidebar:
#     st.header("Instructions")
#     st.write("1. Record a video on your phone (Face-on view).")
#     st.write("2. Upload the file here.")
#     st.write("3. Watch the AI trace your mechanics.")

# uploaded_file = st.file_uploader("Upload your swing (MP4, MOV)", type=['mp4', 'mov', 'avi'])

# if uploaded_file is not None:
#     # Use a temp file to store the upload so OpenCV can read it
#     tfile = tempfile.NamedTemporaryFile(delete=False) 
#     tfile.write(uploaded_file.read())
    
#     video_cap = cv2.VideoCapture(tfile.name)
#     st_frame = st.empty() # Placeholder for the processing loop

#     # Processing loop
#     while video_cap.isOpened():
#         ret, frame = video_cap.read()
#         if not ret:
#             break

#         # Convert to RGB for MediaPipe
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(image)

#         # Draw landmarks
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
#             )

#             # DATA SCIENCE METRIC: Lead Arm Angle
#             # Landmarks: 11 (L Shoulder), 13 (L Elbow), 15 (L Wrist)
#             landmarks = results.pose_landmarks.landmark
#             l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
#             angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            
#             # Display angle on frame
#             cv2.putText(image, f"Arm Angle: {int(angle)}deg", 
#                         tuple(np.multiply(l_elbow, [image.shape[1], image.shape[0]]).astype(int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         # Display the processed frame
#         st_frame.image(image, channels="RGB", use_container_width=True)

#     video_cap.release()
#     st.success("Analysis Complete!")

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculates the angle between three points (b is the vertex)"""
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# 2. Streamlit UI Setup
st.set_page_config(page_title="Golf AI Pro", page_icon="⛳", layout="wide")
st.title("⛳ AI Golf Swing Analyzer")
st.markdown("""
**Data Science Portfolio Project**: This app uses Computer Vision to track joint mechanics.
Once processing is complete, use the video player below to **scrub through your swing** or watch in **slow motion**.
""")

uploaded_file = st.file_uploader("Upload your swing (MP4, MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Save upload to a temp file for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    tfile.close() # Close to allow OpenCV to access
    
    video_cap = cv2.VideoCapture(tfile.name)
    
    # Get original video properties for the new file
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output file setup - using H264 for web compatibility
    output_path = "analyzed_swing.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'H264') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # UI Feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processing loop
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        # AI Analysis
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Draw Skeleton
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
            )

            # Metric Logic
            try:
                landmarks = results.pose_landmarks.landmark
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                # Draw angle on frame
                cv2.putText(image, f"Arm: {int(angle)}deg", 
                            tuple(np.multiply(l_elbow, [width, height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass

        # Save frame to the new video file
        res_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(res_frame)
        
        # Progress math
        current_frame = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress_bar.progress(min(current_frame / frame_count, 1.0))
        status_text.text(f"AI Analyzing: Frame {current_frame} of {frame_count}")

    # Release everything
    video_cap.release()
    out.release()
    status_text.empty()
    progress_bar.empty()

    # 3. Final Output Display
    st.success("✅ Analysis Complete! Scrub through your swing below.")
    
    if os.path.exists(output_path):
        with open(output_path, 'rb') as v_file:
            video_bytes = v_file.read()
            # This player includes the scrubber and speed controls
            st.video(video_bytes, format="video/mp4")
        
        st.download_button(label="Download AI Swing", data=video_bytes, file_name="ai_swing.mp4", mime="video/mp4")
        
        # Clean up temp files
        os.remove(tfile.name)
        os.remove(output_path)
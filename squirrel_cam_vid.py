import time
from datetime import datetime
import os
import cv2
import numpy as np
import subprocess
from picamera2 import Picamera2
from ultralytics import YOLO
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

# --- Configuration ---
TARGET_OBJECTS = ['cat', 'bear', 'person','dog','bird','squirrel','rabbit']
CONFIDENCE_THRESHOLD = 0.5
VIDEO_FOLDER = "squirrel_videos"
DETECTION_INTERVAL_SECONDS = 5
VIDEO_DURATION_SECONDS = 60
COOLDOWN_SECONDS = 30 # Wait this many seconds after recording

def main():
    # Load YOLO model
    try:
        model = YOLO("yolov8n.pt")
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Create the low-resolution configuration for detection captures
    detection_config = picam2.create_still_configuration(main={"size": (640, 480), "format": "XRGB8888"})
    
    # Create the video configuration for recording at 4K resolution
    video_config = picam2.create_video_configuration(main={"size": (3840, 2160)})
    
    # Create the H264Encoder object with a higher bitrate for 4K
    h264_encoder = H264Encoder(20000000) 
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)

    last_detection_time = 0
    
    # Start the camera in the detection configuration initially
    picam2.configure(detection_config)
    picam2.start()

    try:
        print("Detection started. Press Ctrl+C to exit.")
        while True:
            current_time = time.time()
            # Wait for the cooldown period to pass after a video recording
            if (current_time - last_detection_time) < COOLDOWN_SECONDS:
                time.sleep(1) # Small delay to prevent busy-waiting
                continue

            # Capture a low-res image for detection
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Perform Object Detection
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            detected_objects = [model.names[int(box.cls)] for box in results[0].boxes]
            
            if any(obj in TARGET_OBJECTS for obj in detected_objects):
                print("Object of interest detected! Recording video and audio...")

                # Switch to the video configuration
                picam2.switch_mode(video_config)
                
                # Add a delay to allow the camera pipeline to stabilize
                time.sleep(0.5)

                # Record the video
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                video_filename = os.path.join(VIDEO_FOLDER, f"video_{timestamp}.mp4")
                
                # Create the FfmpegOutput object with the video filename
                video_output = FfmpegOutput(video_filename)
                
                # Pass the encoder and the output object to start_recording()
                picam2.start_recording(h264_encoder, video_output)
                
                # Record for the specified duration
                time.sleep(VIDEO_DURATION_SECONDS)
                
                picam2.stop_recording()
                print(f"Recorded {video_filename}")

                # Wait for the audio recording to finish
                #audio_process.wait()
                #print(f"Recorded {audio_filename}")

                # Switch back to the low-res configuration for detection
                picam2.switch_mode(detection_config)
                
                # Update the last detection time to start the cooldown
                last_detection_time = current_time
            
            # Sleep to meet the detection interval
            time_to_next_detection = DETECTION_INTERVAL_SECONDS - (time.time() - current_time)
            if time_to_next_detection > 0:
                time.sleep(time_to_next_detection)

    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()

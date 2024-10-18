from flask import Flask, render_template, request, Response
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")


# Function to generate frames for real-time object detection
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLO detection on the current frame
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame to the webpage in proper format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


# Video feed route for real-time video streaming
@app.route('/video_feed')
def video_feed():
    video_path = "uploaded_video.mp4"  # Use the uploaded video path
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main route for uploading video and rendering index.html
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join("uploaded_video.mp4")
            video_file.save(video_path)
            return render_template('index.html', video_uploaded=True)
    return render_template('index.html', video_uploaded=False)


if __name__ == '__main__':
    app.run(debug=True)

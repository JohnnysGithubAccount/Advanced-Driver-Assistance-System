import json
import math
from flask import Flask, render_template, Response
import datetime
import time
import cv2
from ultralytics import YOLO
import cvzone
import threading
import queue
import torch
from sort import *
from lane_segment_processing import traditional_approach, deep_learning_handle
from utils import traffic_signs, classNames, lane_segmentation, vehicle_heights, average_focal_length


print(torch.cuda.is_available())


# Initialize the Flask app
app = Flask(__name__)

# Load YOLO model
print("INFO: Initializing YOLO model")
yolo_model = YOLO(r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision"
                  r"\Detection System for Autonomous Driving Car"
                  r"\Moving Object Detection\weights\yolo_model7\weights\best.pt")

# Set the video source
# video_source = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision" \
#                r"\Detection System for Autonomous Driving Car" \
#                r"\Moving Object Detection\data\video" \
#                r"\test8.mp4"
video_source = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision" \
               r"\Detection System for Autonomous Driving Car" \
               r"\Moving Object Detection\data\video" \
               r"\project_video.mp4"
# video_source = 0
cap = cv2.VideoCapture(video_source)

vehicle_tracker = Sort(max_age=20, min_hits=3, iou_threshold=.3)
sign_tracker = Sort(max_age=20, min_hits=3, iou_threshold=.3)

danger_region = np.array([
    [300, 720],
    [550, 420+70],
    [730, 420+70],
    [1280-250, 720]
], np.int32)

# either Driving Safely or Collision Warning
driving_situation = "Driving Safely"  # global variable containing the current driving situation

# name: {exist: 0, not_found: 0, num_waits}
frame_counts = {}
list_signs = []
wait_frames_threshold = 2  # wait 5 frames before make sure that is a sign
delete_sign_threshold = 10

# Check if the video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a queue to hold processed frames for YOLO and TFLite
yolo_queue = queue.Queue(maxsize=10)
tflite_queue = queue.Queue(maxsize=10)


def process_yolo(frame):
    results = yolo_model(frame, stream=True, device="0")
    return results


def process_lane(frame, mask=None, approach: str = 'deep'):
    if approach == 'deep':
        weight_mask = deep_learning_handle(mask, frame)
    else:
        weight_mask = traditional_approach(frame)

    return cv2.addWeighted(frame, 1, weight_mask, 0.5, 0)


def process_segmented_lane(frame):
    resized_frame = cv2.resize(frame, (256, 256)) / 255.0

    start_time = time.time()
    segmented_mask = lane_segmentation(resized_frame, isQuantized=True)
    print(f"Lane segmentation inference time: {(time.time() - start_time) * 1_000}ms")

    segmented_mask = segmented_mask > 0.5
    segmented_mask = segmented_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    segmented_mask = cv2.morphologyEx(segmented_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cv2.resize(segmented_mask, (frame.shape[1], frame.shape[0]))


def generate_frames():
    global driving_situation
    global frame_counts
    global list_signs
    collision_detected = False
    num_frames = 0
    to_delete = []

    while True:
        success, img = cap.read()
        if not success:
            break

        # Start threads for both models
        yolo_thread = threading.Thread(target=lambda q, arg1: q.put(process_yolo(arg1)), args=(yolo_queue, img))
        lane_segment_thread = threading.Thread(target=lambda q, arg1: q.put(process_segmented_lane(arg1)),
                                               args=(tflite_queue, img.astype(np.float32)))

        yolo_thread.start()
        lane_segment_thread.start()

        vehicle_detections = np.empty((0, 5))
        sign_detections = np.empty((0, 5))

        # Get results from YOLO queue
        if not yolo_queue.empty():
            yolo_results = yolo_queue.get()
            collision_hist = []
            for r in yolo_results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    cvzone.cornerRect(img,
                                      bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                      colorR=(255, 0, 0))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    check_collision_right = cv2.pointPolygonTest(danger_region, (int(x2), int(y2)), False)
                    check_collision_left = cv2.pointPolygonTest(danger_region, (int(x1), int(y2)), False)

                    if check_collision_right > 0 or check_collision_left > 0:
                        collision_detected = True
                    elif check_collision_right == 0 or check_collision_left == 0:
                        collision_detected = True
                    else:
                        if not collision_detected:
                            collision_detected = False

                    # Check if the detected class is a vehicle
                    if class_name in vehicle_heights:
                        # Calculate the height of the bounding box in pixels
                        bbox_height = h

                        # Use the corresponding height of the vehicle
                        actual_height = vehicle_heights[class_name]

                        # Calculate distance
                        distance = (average_focal_length * actual_height) / bbox_height
                        distance = distance / 10000

                        cvzone.putTextRect(img, f"{class_name} | {distance:.2f} m",
                                           (max(0, int(x1)), max(35, int(y1))),  # bboxes coordinates
                                           scale=1,  # make things smaller
                                           thickness=1,  # letter thickness
                                           offset=2,
                                           colorR=(255, 255, 255),
                                           colorT=(255, 0, 0))

                        if conf > .3:
                            current_array = np.array([
                                int(x1), int(y1),
                                int(x2), int(y2)
                                , conf])

                            vehicle_detections = np.vstack((vehicle_detections, current_array))

                    else:
                        cvzone.putTextRect(img, f"{traffic_signs[class_name]} | {conf}",
                                           (max(0, int(x1)), max(35, int(y1))),  # bboxes coordinates
                                           scale=1,  # make things smaller
                                           thickness=1,  # letter thickness
                                           offset=2,
                                           colorR=(255, 255, 255),
                                           colorT=(255, 0, 0))

                        if conf > .3:
                            current_array = np.array([
                                int(x1), int(y1),
                                int(x2), int(y2)
                                , conf])

                            sign_detections = np.vstack((sign_detections, current_array))

                            if class_name in frame_counts:
                                # name: {exist: 0, not_found: 0, num_waits}
                                frame_counts[class_name][0] += 1
                                print('test')
                                print(frame_counts)
                            else:
                                # name: {exist: 0, not_found: 0, num_waits}
                                frame_counts[class_name] = [1, 0, 0]

            vehicleTracker = vehicle_tracker.update(vehicle_detections)
            signTracker = sign_tracker.update(sign_detections)

            driving_situation = "Collision Warning" if collision_detected else "Driving Safely"

            # check for collision
            for result in vehicleTracker:
                x1, y1, x2, y2, object_id = result
                x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
                cx, cy = int(x1) + w // 2, int(y1) + h // 2
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            # update signs to gui
            for result in signTracker:
                x1, y1, x2, y2, object_id = result
                x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
                cx, cy = int(x1) + w // 2, int(y1) + h // 2
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Get results from lane segmentation queue
        if not tflite_queue.empty():
            segmented_mask = tflite_queue.get()
            mask = segmented_mask * 255

            result = process_lane(img, mask, approach='deep')
            if not isinstance(result, int):
                img = result.copy()
                cv2.polylines(img, [danger_region], isClosed=True, color=(0, 175, 255), thickness=3)
            else:
                cv2.polylines(img, [danger_region], isClosed=True, color=(0 ,175, 255), thickness=3)

        # Prepare the frame for streaming
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        if isinstance(video_source, str):
            if cv2.waitKey(15) == ord('q'):
                break

        print(frame_counts)
        for key, value in frame_counts.items():
            if value[1] >= delete_sign_threshold:
                to_delete.append(key)
                if key in list_signs:
                    list_signs.remove(key)
                continue
            if value[2] == wait_frames_threshold:
                if value[0] < wait_frames_threshold:
                    to_delete.append(key)
                    if key in list_signs:
                        list_signs.remove(key)
                else:
                    if key not in list_signs:
                        list_signs.append(key)
            if value[0] < value[2]:
                value[1] += 1
            value[2] += 1
        for key in to_delete:
            del frame_counts[key]
        to_delete = []

        num_frames += 1

        if len(list_signs) > 0:
            print(f"Frame {num_frames}: {list_signs}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_current_time():
    while True:
        # Format the current time as "weekday, day/month/year hour:minute:second"
        current_time = datetime.datetime.now().strftime('%A, %d/%m/%Y %H:%M:%S')
        yield f"data: {current_time}\n\n"  # Send the current time as Server-Sent Events
        time.sleep(1)  # Update every second


def get_situation():
    global driving_situation
    while True:
        print('Sent:', driving_situation)
        yield f"data: {driving_situation}\n\n"
        time.sleep(1)


def send_signs():
    global list_signs
    while True:
        print('Signs:', list_signs)
        cleaned_signs = [sign.replace('.', '') for sign in list_signs]
        print('Sent:', cleaned_signs)
        yield f"data: {json.dumps(cleaned_signs)}\n\n"
        time.sleep(1)


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/show_time')
def show_time():
    return Response(get_current_time(), mimetype='text/event-stream')


@app.route('/get_driving_situation')
def get_driving_situation():
    return Response(get_situation(), mimetype='text/event-stream')


@app.route('/get_traffic_signs')
def update_signs():
    return Response(send_signs(), mimetype='text/event-stream')


if __name__ == "__main__":
    try:
        # threading.Thread(target=update_driving_situation, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cap.release()  # Release the camera when done

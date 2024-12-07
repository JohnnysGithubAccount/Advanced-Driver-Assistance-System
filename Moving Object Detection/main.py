from ultralytics import YOLO
import cvzone
import cv2
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from keras.models import load_model


lane_segmentation_path = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision" \
                         r"\Detection System for Autonomous Driving Car" \
                         r"\Lane Segmentation\model\seg_model.h5"

loaded_model = load_model(lane_segmentation_path)

lane_segmentation_path = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision" \
                         r"\Detection System for Autonomous Driving Car" \
                         r"\Lane Segmentation\model\quantized_model.tflite"
interpreter = tf.lite.Interpreter(model_path=lane_segmentation_path)


def lane_segmentation(image, isQuantized: bool = False):
    if isQuantized:
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'],
                               tf.expand_dims(image, axis=0))

        # Run the inference
        interpreter.invoke()

        # Get the output tensor
        predicted_mask = interpreter.get_tensor(output_details[0]['index'])[0]

    else:
        predicted_mask = loaded_model.predict(tf.expand_dims(image, axis=0))[0]

    return predicted_mask


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("data/video/test5.mp4")
model = YOLO("weights/best1.pt")

# classNames = ['DP.135', 'P.102', 'P.103a', 'P.103b', 'P.103c',
#               'P.104', 'P.106a', 'P.106b', 'P.107a', 'P.112',
#               'P.115', 'P.117', 'P.123a', 'P.123b', 'P.124a',
#               'P.124b', 'P.124c', 'P.125', 'P.127', 'P.128',
#               'P.130', 'P.131a', 'P.137', 'P.245a', 'R.301c',
#               'R.301d', 'R.301e', 'R.302a', 'R.302b', 'R.303',
#               'R.407a', 'R.409', 'R.425', 'R.434', 'S.509a',
#               'W.201a', 'W.201b', 'W.202a', 'W.202b', 'W.203b',
#               'W.203c', 'W.205a', 'W.205b', 'W.205d', 'W.207a',
#               'W.207b', 'W.207c', 'W.208', 'W.209', 'W.210',
#               'W.219', 'W.221b', 'W.224', 'W.225', 'W.227',
#               'W.233', 'W.235', 'W.245a',
#               'bus', 'car', 'motorbike', 'pedestrian', 'truck']

classNames = ['DP.135', 'P.102', 'P.103a', 'P.103b', 'P.103c',
              'P.104', 'P.106a', 'P.106b', 'P.107a', 'P.112',
              'P.115', 'P.117', 'P.123a', 'P.123b', 'P.124a',
              'P.124b', 'P.124c', 'P.125', 'P.127', 'P.128',
              'P.130', 'P.131a', 'P.137', 'P.245a', 'P106b',
              'R.301c', 'R.301d', 'R.301e', 'R.302a', 'R.302b',
              'R.303', 'R.407a', 'R.409', 'R.425', 'R.434', 'R434',
              'S.509a', 'W.201a', 'W.201b', 'W.202a', 'W.202b',
              'W.203b', 'W.203c', 'W.205a', 'W.205b', 'W.205d',
              'W.207a', 'W.207b', 'W.207c', 'W.208', 'W.209',
              'W.210', 'W.219', 'W.221b', 'W.224', 'W.225',
              'W.227', 'W.233', 'W.235', 'W.245a',
              'bus', 'car', 'motorbike', 'pedestrian', 'truck']

traffic_signs = {
    'DP.135': 'End of all previously signed prohibitions',
    'P.102': 'No entry',
    'P.103a': 'No cars',
    'P.103b': 'No right turn for cars',
    'P.103c': 'No left turn for cars',
    'P.104': 'No motorcycles',
    'P.106a': 'No trucks',
    'P.106b': 'No trucks above designated weight',
    'P.107a': 'No buses',
    'P.112': 'No pedestrians',
    'P.115': 'Vehicle weight limit',
    'P.117': 'Height limit',
    'P.123a': 'No left turn',
    'P.123b': 'No right turn',
    'P.124a': 'No left U-turn',
    'P.124b': 'No right U-turn',
    'P.124c': 'No left turn or U-turn',
    'P.125': 'No overtaking',  # No Car Passing
    'P.127': 'Maximum speed limit',
    'P.130': 'No stopping and parking',
    'P.131a': 'No parking',
    'P.137': 'No left or right turn',
    'P.245a': 'Go slow',
    'R.301c': 'Only turning left',
    'R.301d': 'Only turning right',
    'R.301e': 'Only turning left',
    'R.302a': 'Go right to avoid obstacle',
    'R.302b': 'Go left to avoid obstacle',
    'R.303': 'Roundabout',
    'R.407a': 'One way road',
    'R.409': 'U-turn',
    'R.425': 'Hospital',
    'R.434': 'Bus Stop',
    'S.509a': 'Safe height',
    'W.201a': 'Danger-warning-Curve to left',
    'W.201b': 'Danger-warning-Curve to right',
    'W.202a': 'Danger-warning-Multiple left turn',
    'W.202b': 'Danger-warning-Multiple right turn',
    'W.203b': 'Danger-warning-Road narrows ahead on the left side',
    'W.203c': 'Danger-warning-Road narrows ahead on the right side',
    'W.205a': 'Danger-warning-Crossroad',
    'W.205b': 'Danger-warning-Side road junction on the right',
    'W.205d': 'Danger-warning-Side road junction on the left',
    'W.207a': 'Danger-warning-Intersection ahead',
    'W.207b': 'Danger-warning-Intersection ahead on the right',
    'W.207c': 'Danger-warning-Intersection ahead on the left',
    'W.208': 'Danger-warning-Yield to priority road',
    'W.209': 'Danger-warning-Usually have accident ',
    'W.210': 'Danger-warning-Rail road crossing',
    'W.219': 'Danger-warning-Dangerous slope',
    'W.221b': 'Danger-warning-Bumpy road',
    'W.224': 'Danger-warning-Pedestrian crossing',
    'W.225': 'Danger-warning-Children',
    'W.227': 'Danger-warning-Construction site',
    'W.233': 'Danger-warning-Other dangers',
    'W.235': 'Danger-warning-Dual Carriageway',
    'W.245a': 'Danger-warning-Go slow',
    'bus': 'Bus', 'car': 'Car', 'motorbike': 'Motorbike', 'pedestrian': 'Pedestrian', 'truck': 'Truck'
}

vehicle_heights = {
    'car': 1600,
    'bus': 3600,
    'motorbike': 1600,
    'truck': 3300,
    'pedestrian': 1650
}

f_x = 1887.6598193189468
f_y = 1870.0813223886414
average_focal_length = (f_x + f_y) / 2

while True:
    success, img = cap.read()

    if not success:
        break

    original_shape = img.shape

    results = model(img, stream=True, device="0")

    # Set the input tensor
    temp_img = img.copy()
    temp_img = cv2.resize(temp_img, (256, 256)) / 255.0
    temp_img = temp_img.astype(np.float32)

    start_time = time.time()
    mask = lane_segmentation(temp_img, isQuantized=True)
    print(f"Elapsed time: {time.time() - start_time}s")
    mask = mask > 0.5
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow("mask", mask * 255.0)
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cvzone.cornerRect(img, bbox=(x, y, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

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

            else:
                cvzone.putTextRect(img, f"{traffic_signs[class_name]} | {conf}",
                                   (max(0, int(x1)), max(35, int(y1))),  # bboxes coordinates
                                   scale=1,  # make things smaller
                                   thickness=1,  # letter thickness
                                   offset=2,
                                   colorR=(255, 255, 255),
                                   colorT=(255, 0, 0))

        # if cv2.waitKey(1) == ord('q'):
        #     break

    result = img.copy()
    mask = mask.astype(bool)
    result[:, :, 0][mask] = 0
    result[:, :, 1][mask] = 255
    result[:, :, 2][mask] = 0

    # Convert the mask to a 3-channel image for blending
    colored_overlay = np.zeros_like(img)
    colored_overlay[:, :, 0][mask] = 0  # Blue channel
    colored_overlay[:, :, 1][mask] = 255  # Green channel
    colored_overlay[:, :, 2][mask] = 0  # Red channel

    # Define alpha for blending (0.0 = fully transparent, 1.0 = fully opaque)
    alpha = 0.3  # Adjust this value for desired opacity

    # Blend the original image and the overlay
    result = cv2.addWeighted(img, 1 - alpha, colored_overlay, alpha, 0)

    cv2.imshow('Result', result/255.0)
    cv2.waitKey(1)

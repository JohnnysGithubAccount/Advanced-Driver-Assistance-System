# Advanced Driver Assistance System (ADAS)

## Overview

The Advanced Driver Assistance System (ADAS) is a comprehensive solution designed to enhance vehicle safety and facilitate a smoother driving experience. This project incorporates a Flask web application that provides a graphical user interface (GUI) for real-time monitoring and interaction. Key features include distance measurements, collision warning, vehicle and traffic sign detection, and lane segmentation.

## Features
![image](https://github.com/user-attachments/assets/494cd781-9227-40cd-988b-348354e26a71)

- **Vehicle Detection**: Identifies vehicles on the road using advanced object detection algorithms (YOLOv8 and SORT).
- **Distance Measurements**: Quite accurate measurement of the distance between vehicles using computer vision techniques.
- **Collision Warning**: Real-time alerts to prevent potential collisions based on proximity and speed analysis.
- **Traffic Sign Detection**: Recognizes and interprets traffic signs to provide relevant warnings and information to the driver.
- **Lane Segmentation**: Analyzes lane markings for better lane-keeping assistance.
- **Flask Web App GUI**: A user-friendly interface for visualizing data and system status.

## Software Architecture
![image](https://github.com/user-attachments/assets/0b69d5c0-81ad-4b8c-8d81-636d98d8801a)

The system contains 4 main blocks:
- **YOLOv8n**: Perform object detection
- **U-Net**: Lane segmentation
- **Processing**: Processing output of the first 2 blocks the give the final output frame
- **Graphic User Interface**

## Distance estimation
![image](https://github.com/user-attachments/assets/b20c993f-928d-4c59-95f5-af774f585be4)

Too keep things simple, since I have no stereo camera, this method will give averagely precise distance estimation for the system. With the given height of the object in real life will be average height I can find online.

## Lane Segmentation
![image](https://github.com/user-attachments/assets/62399911-4ea6-42ff-9127-488bd27194a2)

Here is the algorithm used for the lane segmentation traditionally which is with out using deep learning, but I replace the original thresholding steps with U-Net for getting the binary map.

## Collision Warning
![image](https://github.com/user-attachments/assets/b4468c9c-e240-4726-b823-f404a9c31196)


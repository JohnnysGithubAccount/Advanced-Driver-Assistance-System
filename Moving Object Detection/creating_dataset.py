import os
import cv2
import time

# Set the paths
video_path = "data/video/data.mp4"
output_dir = "data/images"

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_frame_count = 0  # Count of saved frames

while True:
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("End of video or unable to read the frame.")
        break

    # Save the current frame every 30 frames
    if frame_count % 30 == 0:
        output_path = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")  # Save as JPG
        frame = cv2.resize(frame, (512, 512))
        cv2.imwrite(output_path, frame)
        saved_frame_count += 1  # Increment saved frame count

    # Display the current frame
    cv2.imshow('frame', frame)

    # Increment the frame count
    frame_count += 1

    # Wait for a short duration (1 millisecond) or exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

print(f"Frame extraction complete! {saved_frame_count} frames saved.")
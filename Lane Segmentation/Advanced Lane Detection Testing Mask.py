import cv2
import numpy as np
from matplotlib import pyplot as plt


height, width = 720, 1280
# Load your original image and the binary mask
image = cv2.imread('images/mask.png')  # Original image
image = cv2.resize(image, (1280, 720))
binary_mask = cv2.imread('images/mask.png', cv2.IMREAD_GRAYSCALE)  # Binary mask
binary_mask = cv2.resize(binary_mask, (1280, 720))

# Apply a filter mask to only keep the lower half
filter_mask = np.concatenate((np.zeros((360, 1280)), np.ones((360, 1280))), axis=0).astype(np.uint8)
binary_mask = binary_mask * filter_mask

# Define the points corresponding to the corners for perspective transformation
src_points = np.float32([
    [200, 720],
    [550, 440],
    [700, 440],
    [1280-180, 720]
])

# Define the destination points for the bird's-eye view
dst_points = np.float32([
    [0, height],        # Bottom left
    [0, 0],             # Top left
    [width, 0],         # Top right
    [width, height]     # Bottom right
])

matrix = cv2.getPerspectiveTransform(src_points, dst_points)
bird_eye_view = cv2.warpPerspective(binary_mask, matrix, (width, height))

contours, _ = cv2.findContours(bird_eye_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area to keep only lane lines
lane_lines = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # Adjust this threshold based on your needs
        lane_lines.append(contour)

print(len(lane_lines))

lane_line_image = np.zeros_like(image)
cv2.drawContours(lane_line_image, lane_lines, -1, (0, 255, 0), 3)  # Green color for lane lines

# Find the starting point using histogram
histogram = np.sum(bird_eye_view[bird_eye_view.shape[0] // 2:, :], axis=0)
midpoint = int(histogram.shape[0] / 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Set parameters for sliding windows
n_windows = 20
window_height = int(height / n_windows)
margin = 100
min_pix = 50

# Identify the x and y positions of all non-zero pixels in the mask
nonzero = bird_eye_view.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Create an empty list to hold the lane line indices
left_lane_inds = []
right_lane_inds = []

# Current positions of the left and right line
left_current = leftx_base
right_current = rightx_base

for window in range(n_windows):
    # Identify the window boundaries in x and y (both left and right)
    win_y_low = height - (window + 1) * window_height
    win_y_high = height - window * window_height
    win_xleft_low = left_current - margin
    win_xleft_high = left_current + margin
    win_xright_low = right_current - margin
    win_xright_high = right_current + margin

    # Draw the windows on the visualization image
    cv2.rectangle(lane_line_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(lane_line_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    cv2.imshow('windows', lane_line_image)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high) &
                      (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
    good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <= win_xright_high) &
                       (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]

    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # If you found more than min_pix pixels, recenter the next window on their mean position
    if len(good_left_inds) > min_pix:
        left_current = int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > min_pix:
        right_current = int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract the pixel positions for the left and right lanes
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

try:
    print(lefty, leftx, righty, rightx)
    print(lefty.shape, leftx.shape, righty.shape, rightx.shape)
    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Print the beginning and end of the lane lines
    print(f"Left Lane Start: (x: {int(left_fitx[0])}, y: {int(ploty[0])})")
    print(f"Left Lane End: (x: {int(left_fitx[-1])}, y: {int(ploty[-1])})")
    print(f"Right Lane Start: (x: {int(right_fitx[0])}, y: {int(ploty[0])})")
    print(f"Right Lane End: (x: {int(right_fitx[-1])}, y: {int(ploty[-1])})")

    # Create an image to draw the lane lines
    lane_line_image = np.zeros_like(image)
    for y in range(len(ploty)):
        cv2.circle(lane_line_image, (int(left_fitx[y]), int(ploty[y])), 20, (0, 0, 0), -1)  # Left lane
        cv2.circle(lane_line_image, (int(right_fitx[y]), int(ploty[y])), 20, (0, 0, 0), -1)  # Right lane

    # Fill the area between the lane lines using the fitted points
    vertices = np.array(list(zip(np.concatenate((left_fitx, right_fitx[::-1])),
                                  np.concatenate((ploty, ploty[::-1])))), dtype=np.int32)

    # Reshape vertices to the required format for fillPoly
    # vertices = vertices.reshape((-1, 1, 2))

    cv2.imshow("lane_line_image2", lane_line_image)
    cv2.fillPoly(lane_line_image, [vertices], (0, 255, 0))
    cv2.imshow('filled', lane_line_image)
    # Unwrap the lane_line_image back to the original perspective
    unwarp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    lane_line_image_original = cv2.warpPerspective(lane_line_image, unwarp_matrix, (bird_eye_view.shape[1], bird_eye_view.shape[0]))

    # Blend the unwrapped lane line image with the original image
    result = cv2.addWeighted(image, 1, lane_line_image_original, 0.5, 0)
    cv2.imshow('result', result)
    for i, point in enumerate(src_points):
        cv2.circle(bird_eye_view, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)  # Red color

    # Display all stages of transformation
    cv2.imshow('Original Frame', bird_eye_view)
except:
    for i, point in enumerate(src_points):
        cv2.circle(bird_eye_view, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)  # Red color

    cv2.imshow('Bird Eye View', bird_eye_view)

"""
    [250, 720],
    [600, 420+70],
    [730, 420+70],
    [1280-250, 720]
"""
# Draw source points on the original image
for point in src_points:
    cv2.circle(image, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Draw green circles
cv2.imshow('Mask', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

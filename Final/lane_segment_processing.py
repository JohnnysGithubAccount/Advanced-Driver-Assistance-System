import numpy as np
import matplotlib.pyplot as plt
import cv2


height, width = 720, 1280

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

# Set parameters for sliding windows
n_windows = 20
window_height = int(height / n_windows)
margin = 100
min_pix = 50


def traditional_approach(frame):
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # Apply the perspective transformation to get the bird's-eye view
    bird_eye_view = cv2.warpPerspective(frame, matrix, (width, height))

    # Convert to HLS color space
    hls = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2HLS)

    # Define color range for lane lines in HLS
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Threshold the HLS image to get only lane line colors
    mask = cv2.inRange(hls, lower_white, upper_white)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Opening
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)  # Closing

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to keep only lane lines
    lane_lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust this threshold based on your needs
            lane_lines.append(contour)

    # Create a blank image to draw the lane lines
    lane_line_image = np.zeros_like(bird_eye_view)
    cv2.drawContours(lane_line_image, lane_lines, -1, (0, 255, 0), 3)  # Green color for lane lines

    # Combine the lane line image with the bird's-eye view
    combined_image = cv2.addWeighted(bird_eye_view, 1, lane_line_image, 0.5, 0)

    # Find the starting point using histogram
    histogram = np.sum(mask_close[mask_close.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Identify the x and y positions of all non-zero pixels in the mask
    nonzero = mask_close.nonzero()
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

    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Print the beginning and end of the lane lines
    print(f"Left Lane Start: (x: {int(left_fitx[0])}, y: {int(ploty[0])})")
    print(f"Left Lane End: (x: {int(left_fitx[-1])}, y: {int(ploty[-1])})")
    print(f"Right Lane Start: (x: {int(right_fitx[0])}, y: {int(ploty[0])})")
    print(f"Right Lane End: (x: {int(right_fitx[-1])}, y: {int(ploty[-1])})")

    # Create an image to draw the lane lines
    lane_line_image = np.zeros_like(bird_eye_view)
    for y in range(len(ploty)):
        cv2.circle(lane_line_image, (int(left_fitx[y]), int(ploty[y])), 5, (255, 0, 0), -1)  # Left lane
        cv2.circle(lane_line_image, (int(right_fitx[y]), int(ploty[y])), 5, (0, 0, 255), -1)  # Right lane

    # Fill the area between the lane lines using the fitted points
    vertices = np.array(list(zip(np.concatenate((left_fitx, right_fitx[::-1])),
                                 np.concatenate((ploty, ploty[::-1])))), dtype=np.int32)

    cv2.fillPoly(lane_line_image, [vertices], (0, 255, 0))

    # Unwrap the lane_line_image back to the original perspective
    unwarp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    lane_line_image_original = cv2.warpPerspective(lane_line_image, unwarp_matrix, (frame.shape[1], frame.shape[0]))

    return lane_line_image_original


def deep_learning_handle(predicted_mask, image, isDebugging: bool = False, video_source: int = 3):
    if video_source == 3:
        src_points = np.float32([
            [330, 720],
            [580, 420 + 100],
            [700, 420 + 100],
            [1280 - 370, 720]
        ])
    elif video_source == 1:
        src_points = np.float32([
            [0, 500],     # Bottom left
            [550, 300],   # Top left
            [900, 300],   # Top right
            [1280, 550]   # Bottom right
        ])
    else:
        pass

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    if isDebugging:
        cv2.imshow("received frame", image)
    rows = image.shape[0]
    columns = image.shape[1]
    filter_mask = np.concatenate((np.zeros((int(rows/2), columns)),
                                  np.ones((int(rows/2), columns))), axis=0).astype(np.uint8)
    predicted_mask = predicted_mask * filter_mask
    bird_eye_view = cv2.warpPerspective(predicted_mask, matrix, (width, height))
    if isDebugging:
        cv2.imshow("Bird eye view mask", bird_eye_view)
    contours, _ = cv2.findContours(bird_eye_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if isDebugging:
        # Create a blank image to draw contours
        contour_image = np.zeros_like(bird_eye_view)

        # Draw contours
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2
        cv2.imshow(' Displaying contours', contour_image)

    # Filter contours based on area to keep only lane lines
    lane_lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust this threshold based on your needs
            lane_lines.append(contour)
    if isDebugging:
        print(f"Number of lane line filtered: {len(lane_lines)}")
    lane_line_image = np.zeros_like(image)
    cv2.drawContours(lane_line_image, lane_lines, -1, (0, 255, 0), 3)  # Green color for lane lines
    # Find the starting point using histogram
    histogram = np.sum(bird_eye_view[bird_eye_view.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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

        if isDebugging:
            # Draw the windows on the visualization image
            cv2.rectangle(lane_line_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(lane_line_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            cv2.imshow('lane_line_image_check', lane_line_image)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high) &
                          (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <= win_xright_high) &
                           (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]

        if isDebugging:
            print(good_right_inds.shape)
            print(good_left_inds.shape)

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

    # Fit a second order polynomial to each lane
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if isDebugging:
            # Print the beginning and end of the lane lines
            print(f"Left Lane Start: (x: {int(left_fitx[0])}, y: {int(ploty[0])})")
            print(f"Left Lane End: (x: {int(left_fitx[-1])}, y: {int(ploty[-1])})")
            print(f"Right Lane Start: (x: {int(right_fitx[0])}, y: {int(ploty[0])})")
            print(f"Right Lane End: (x: {int(right_fitx[-1])}, y: {int(ploty[-1])})")

        # Create an image to draw the lane lines
        lane_line_image = np.zeros_like(image)
        for y in range(len(ploty)):
            cv2.circle(lane_line_image, (int(left_fitx[y]), int(ploty[y])), 5, (255, 0, 0), -1)  # Left lane
            cv2.circle(lane_line_image, (int(right_fitx[y]), int(ploty[y])), 5, (0, 0, 255), -1)  # Right lane

        # Fill the area between the lane lines using the fitted points
        vertices = np.array(list(zip(np.concatenate((left_fitx, right_fitx[::-1])),
                                     np.concatenate((ploty, ploty[::-1])))), dtype=np.int32)

        # cv2.imshow("lane_line_image2", lane_line_image)
        # cv2.imshow("original", image)
        cv2.fillPoly(lane_line_image, [vertices], (0, 255, 0))
        # cv2.imshow('filled', lane_line_image)

        # Unwrap the lane_line_image back to the original perspective
        unwarp_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        lane_line_image_original = cv2.warpPerspective(lane_line_image, unwarp_matrix,
                                                       (bird_eye_view.shape[1], bird_eye_view.shape[0]))

        # final_result = cv2.addWeighted(image, 1, lane_line_image_original, 0.5, 0)
        final_result = np.clip(cv2.addWeighted(image, 1, lane_line_image_original, 0.5, 0), 0, 255).astype('uint8')
        final_result = cv2.subtract(final_result, image)
        if isDebugging:
            cv2.imshow("Output checking", final_result)
        return final_result
    except Exception as e:
        print(e)
        print("Fail lane detection frame")
        return cv2.subtract(image, image)


def main():
    pass


if __name__ == "__main__":
    main()

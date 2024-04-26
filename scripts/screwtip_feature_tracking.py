import cv2
import numpy as np

# Load the series of images
images = [cv2.imread(f'image_{i}.jpg') for i in range(num_images)]

# Initialize the tracker
tracker = cv2.MultiTracker_create()

# Iterate through the images
for i, image in enumerate(images):
    # Detect the screw tip in the first frame
    if i == 0:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the contour with the largest area (assuming this is the screw)
            largest_contour = max(contours, key=cv2.contourArea)

            # Find the tip of the screw
            M = cv2.moments(largest_contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            tip_x, tip_y = cv2.pointPolygonTest(largest_contour, (cx, cy), True)

            # Add the screw tip to the tracker
            tracker.add(cv2.MultiTracker_create(), image, (tip_x, tip_y, 10, 10))
    else:
        # Update the tracker and get the new position of the screw tip
        success, boxes = tracker.update(image)
        for i, newbox in enumerate(boxes):
            x, y, w, h = [int(v) for v in newbox]
            print(f'Screw tip position in frame {i}: ({x}, {y})')

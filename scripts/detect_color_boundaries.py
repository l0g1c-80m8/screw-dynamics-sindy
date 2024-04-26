import cv2
import numpy as np

images = [
    r'./data/Screwdriving Model Experiments/14_60_5_M5_1500/camera/c_1713917273904.680176.png',
    r'./data/Screwdriving Model Experiments/14_60_5_M5_1500/camera/c_1713917266711.516113.png',
    r'./data/Screwdriving Model Experiments/12_60_3_M4_2500/camera/c_1713916523801.533691.png',
    r'./data/Screwdriving Model Experiments/5_30_5_M5_1500/camera/c_1713834305043.865967.png',
    r'./data/Screwdriving Model Experiments/1_30_1_M4_500/camera/c_1713832807239.558105.png',
    r'./data/Screwdriving Model Experiments/7_30_7_M6_500/camera/c_1713835482379.481689.png',
]


# Function to track mouse events
def mouse_callback(event, x, y, flags, param):
    global hsv_img, lower_bound, upper_bound

    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = hsv_img[y, x]
        lower_bound = np.array([pixel_value[0] - 10, 100, 100])
        upper_bound = np.array([pixel_value[0] + 10, 255, 255])
        print("Lower Bound: ", lower_bound)
        print("Upper Bound: ", upper_bound)
        print("Press 'q' to quit")
        update_sliders()


# Function to update sliders
def update_sliders():
    cv2.setTrackbarPos('Hue Lower', 'HSV Image', lower_bound[0])
    cv2.setTrackbarPos('Hue Upper', 'HSV Image', upper_bound[0])
    cv2.setTrackbarPos('Saturation Lower', 'HSV Image', lower_bound[1])
    cv2.setTrackbarPos('Saturation Upper', 'HSV Image', upper_bound[1])
    cv2.setTrackbarPos('Value Lower', 'HSV Image', lower_bound[2])
    cv2.setTrackbarPos('Value Upper', 'HSV Image', upper_bound[2])


# Callback function for trackbar
def on_trackbar(val):
    global lower_bound, upper_bound
    lower_bound[0] = cv2.getTrackbarPos('Hue Lower', 'HSV Image')
    upper_bound[0] = cv2.getTrackbarPos('Hue Upper', 'HSV Image')
    lower_bound[1] = cv2.getTrackbarPos('Saturation Lower', 'HSV Image')
    upper_bound[1] = cv2.getTrackbarPos('Saturation Upper', 'HSV Image')
    lower_bound[2] = cv2.getTrackbarPos('Value Lower', 'HSV Image')
    upper_bound[2] = cv2.getTrackbarPos('Value Upper', 'HSV Image')
    print("Lower Bound: ", lower_bound)
    print("Upper Bound: ", upper_bound)


# Load an image
image = cv2.imread(images[-1])
resized_img = cv2.resize(image, (800, 600))

# Convert the image to HSV color space
hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

# Initialize lower and upper bounds
lower_bound = np.array([0, 100, 100])
upper_bound = np.array([255, 255, 255])

# Create a window to display the image
cv2.namedWindow('HSV Image')

# Create trackbars for color selection
cv2.createTrackbar('Hue Lower', 'HSV Image', lower_bound[0], 179, on_trackbar)
cv2.createTrackbar('Hue Upper', 'HSV Image', upper_bound[0], 179, on_trackbar)
cv2.createTrackbar('Saturation Lower', 'HSV Image', lower_bound[1], 255, on_trackbar)
cv2.createTrackbar('Saturation Upper', 'HSV Image', upper_bound[1], 255, on_trackbar)
cv2.createTrackbar('Value Lower', 'HSV Image', lower_bound[2], 255, on_trackbar)
cv2.createTrackbar('Value Upper', 'HSV Image', upper_bound[2], 255, on_trackbar)

cv2.setMouseCallback('HSV Image', mouse_callback)

while True:
    # Create a mask using the lower and upper bounds
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(resized_img, resized_img, mask=mask)

    # Display the masked image
    cv2.imshow('HSV Image', masked_image)

    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()


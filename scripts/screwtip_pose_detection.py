import os
import cv2
import numpy as np
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data',
                        action='store', dest='data_dir', help='data directory')

    return parser.parse_args()


def get_pose(filepath):
    image = cv2.imread(filepath)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([110, 20, 50])
    upper_red = np.array([179, 120, 100])

    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    threshold_area = 10
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

    mask_filled = np.zeros_like(mask)
    cv2.drawContours(mask_filled, filtered_contours, -1, 255, thickness=cv2.FILLED)

    # Get centroid of the filled contour
    centroid = None
    for cnt in filtered_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            centroid = (centroid_x, centroid_y)
            break  # Assuming there's only one contour

    if centroid is not None:
        cv2.circle(image, centroid, 1, (0, 0, 255), -1)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    filepath = os.path.join(args.data_dir, 'sample_image.png')
    get_pose(filepath)


if __name__ == '__main__':
    args = get_args()
    main()

import os
import re
import cv2
import numpy as np
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data',
                        action='store', dest='data_dir', help='data directory')
    parser.add_argument('--debug', type=bool, default=True,
                        action='store', dest='debug', help='debug images?')

    return parser.parse_args()


def get_pose(filepath):
    image = cv2.imread(filepath)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([110, 20, 50])
    upper_red = np.array([179, 120, 100])

    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    threshold_area = 0
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

    print('in file {}, found {} matching spots '.format(filepath, len(filtered_contours)), end='')

    if filtered_contours:
        largest_contour = max(filtered_contours, key=cv2.contourArea)

        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Get centroid of the filled contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            centroid = (centroid_x, centroid_y)

            print(', and centroid is {}'.format(centroid))

            cv2.circle(image, centroid, 1, (0, 0, 255), -1)

            if args.debug:
                if not 330 <= centroid[0] <=350 and not 225 <= centroid[1] <=245:
                    cv2.imshow('Result', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


def main():
    for item in os.listdir(args.data_dir):
        if not re.match(r'^c_\d+.\d+\.png$', item):
            continue
        filepath = os.path.join(args.data_dir, item)
        get_pose(filepath)


if __name__ == '__main__':
    args = get_args()
    main()

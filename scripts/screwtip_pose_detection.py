import os
import sys
import re
import cv2
import numpy as np
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data',
                        action='store', dest='data_dir', help='data directory')
    parser.add_argument('--out_dir', type=str, default='./data',
                        action='store', dest='out_dir', help='output directory')
    parser.add_argument('--out_file', type=str, default='observation.csv',
                        action='store', dest='out_file', help='output file')
    parser.add_argument('--debug', type=bool, default=False,
                        action='store', dest='debug', help='debug images?')

    return parser.parse_args()


def get_pixel(filepath):
    image = cv2.imread(filepath)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([110, 20, 50])
    upper_red = np.array([179, 120, 100])

    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    threshold_area = 0
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

    # print('in file {}, found {} matching spots'.format(filepath, len(filtered_contours)), end='')

    if filtered_contours:

        height, width = image.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # Initialize variables for closest contour and its distance to the center
        closest_contour = None
        min_distance = float('inf')

        # Iterate through contours
        for contour in contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Compute the distance between centroid and center of the image
            distance = np.sqrt((cX - center_x) ** 2 + (cY - center_y) ** 2)

            # Update closest contour if the distance is smaller
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

        mask_filled = np.zeros_like(mask)
        green_mask = np.zeros_like(image)
        green_mask[:, :] = (0, 255, 0)
        cv2.drawContours(mask_filled, [closest_contour], -1, 255, thickness=cv2.FILLED)

        # Get centroid of the filled contour
        M = cv2.moments(closest_contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            centroid = (centroid_x, centroid_y)

            # print(', and centroid is {}'.format(centroid))

            result = np.where(mask[:, :, np.newaxis] != 0, green_mask, image)
            cv2.circle(result, centroid, 1, (0, 0, 255), -1)

            if args.debug:
                if not 330 <= centroid[0] <= 350 and not 225 <= centroid[1] <= 245:
                    cv2.imshow('Result', result)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            return centroid


def get_depth(filepath, pixel):
    depth_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    x, y = pixel
    block = depth_image[max(0, y - 2):min(depth_image.shape[0], y + 2), max(0, x - 2):min(depth_image.shape[1], x + 2)]
    average_depth = np.mean(block)
    return average_depth / 1000.


def write_to_file(data, filepath):
    # Header for your data
    headers = ['time', 'X', 'Y', 'Z']
    data_with_headers = np.vstack((headers, data))
    np.savetxt(filepath, data_with_headers, delimiter=",", fmt="%s")


def fix_data(data):
    zero_indices = np.where(data[:, -1] == 0.0)[0]
    for zero_index in zero_indices:
        data[zero_index, -1] = (
                np.sum(data[zero_index - 4:zero_index, -1]) +
                np.sum(data[zero_index:zero_index + 4, -1]) /
                8.
        )


def pixel_to_3d(z_depth, cx, cy):
    # Example usage
    u = 320
    v = 240
    fx = 500
    fy = 500

    x = (u - cx) * z_depth / fx
    y = (v - cy) * z_depth / fy
    return x, y, z_depth


def main():
    for subdir in os.listdir(args.data_dir):
        if not re.match(r'^\d{1,2}_\d{2}_\d{1,2}_M\d_\d{3,4}$', subdir):
            continue

        subdir_path = os.path.join(args.data_dir, subdir)
        image_path = os.path.join(subdir_path, 'camera')

        data = []
        for item in os.listdir(image_path):
            if not re.match(r'^c_\d+.\d+\.png$', item):
                continue
            filepath = os.path.join(image_path, item)
            pixel = get_pixel(filepath)
            depth = get_depth(os.path.join(image_path, item.replace('c_', 'd_')), pixel)
            if depth <= .0 and args.debug:
                print('depth {}, for file {}'.format(depth, item))
            data.append((float(item.replace('c_', '').replace('.png', '')), *pixel, depth))

        data = np.array(data)

        fix_data(data)
        write_to_file(data, os.path.join(subdir_path, args.out_file))

        # if args.debug:
        #     print(data)


if __name__ == '__main__':
    args = get_args()
    main()
    sys.exit(0)

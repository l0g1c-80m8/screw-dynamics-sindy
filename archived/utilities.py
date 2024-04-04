import torch
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from math import atan2

def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



def scaled_image(image, output_size = (128,128), save_flag = False):
    """_summary_

    Args:
        image (_type_): _description_ should be an OpenCV format frame object from Realsense
        output_size (tuple, optional): _description_. Defaults to (128,128).
        rgb_scale (int, optional): _description_. Defaults to 255.

    Returns:
        _type_: _description_
    """
    ####Implement Scaling Code Here###
    new_image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    if(save_flag):
        print("Saving Image")
        current_timestamp = datetime.datetime.timestamp(datetime.datetime.now())
        cv2.imwrite(str(current_timestamp)+".jpg", new_image)
        # cv2.imshow("Image", new_image)
        # cv2.waitKey(0)
    return new_image


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


def transposed_conv2d(in_planes, out_planes):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        torch.nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes):
    return torch.nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def torch_binom(n, k):
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask



# def fit_ellipse(X,y):

#     major_axis = 0
#     minor_axis = 0

#     X = np.reshape(X,[len(X),1])
#     y = np.reshape(y,[len(y),1])

#     A = np.hstack([X**2, X*y,y**2,X,y])
#     b = np.ones_like(X)
#     print(X.shape)
#     print(y.shape)
#     x = np.linalg.lstsq(A,b)[0].squeeze()

#     print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))


#     # # Plot the original ellipse from which the data was generated
#     # phi = np.linspace(0, 2*np.pi, X.shape[0]).reshape((X.shape[0],1))
#     # c = np.hstack([np.cos(phi), np.sin(phi)])
#     # ground_truth_ellipse = c.dot(b)
#     # plt.plot(ground_truth_ellipse[:,0], ground_truth_ellipse[:,1], 'k--', label='Generating Ellipse')


#     # Plot the noisy data
#     plt.scatter(X, y, label='Data Points')

#     # Plot the least squares ellipse
#     X_coord, Y_coord = np.meshgrid(X, y)
#     Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
#     plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

#     plt.legend()
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()
#     return



def fit_ellipse_skimage(X1,y1, display_flag = False):
    [x_mean, x_std] = find_mean_std_dev(X1)
    [y_mean, y_std] = find_mean_std_dev(y1)

    new_x = X1[np.where(np.logical_and(X1>=(x_mean-x_std), X1<=(x_mean+x_std)))]
    new_y = y1[np.where(np.logical_and(y1>=(y_mean-y_std), y1<=(y_mean+y_std)))]
    points = np.column_stack((new_x,new_y))
    
    x = points[:,0]
    y = points[:,1]
    ell = EllipseModel()
    ell.estimate(points)
    
    xc,yc,a,b,theta = ell.params

    if(display_flag):
        print("Ellipse Params: ", ell.params)
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        axs[0].scatter(x,y)

        axs[1].scatter(x, y)
        axs[1].scatter(xc, yc, color='red', s=100)
        axs[1].set_xlim(x.min(), x.max())
        axs[1].set_ylim(y.min(), y.max())

        ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none')

        axs[1].add_patch(ell_patch)
        plt.show()
        
    return [xc,yc,a,b,theta]


# def __fit_ellipse(x, y):
#     x, y = x[:, np.newaxis], y[:, np.newaxis]
#     D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
#     S, C = np.dot(D.T, D), np.zeros([6, 6])
#     C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
#     U, s, V = np.linalg.svd(np.dot(np.linalg.inv(S), C))
#     a = U[:, 0]
#     return a

# def ellipse_center(a):
#     b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
#     num = b * b - a * c
#     x0 = (c * d - b * f) / num
#     y0 = (a * f - b * d) / num
#     return np.array([x0, y0])

# def ellipse_axis_length(a):
#     b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
#     up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
#     down1 = (b * b - a * c) * (
#         (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
#     )
#     down2 = (b * b - a * c) * (
#         (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
#     )
#     print("d1: ", down1," d2: ", down2)
#     res1 = np.sqrt(up / down1)
#     res2 = np.sqrt(up / down2)
#     return np.array([res1, res2])

# def ellipse_angle_of_rotation(a):
#     b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
#     return atan2(2 * b, (a - c)) / 2

# def fit_ellipse(x, y):
#     """@brief fit an ellipse to supplied data points: the 5 params
#         returned are:
#         M - major axis length
#         m - minor axis length
#         cx - ellipse centre (x coord.)
#         cy - ellipse centre (y coord.)
#         phi - rotation angle of ellipse bounding box
#     @param x first coordinate of points to fit (array)
#     @param y second coord. of points to fit (array)
#     """
#     [x_mean, x_std] = find_mean_std_dev(x)
#     [y_mean, y_std] = find_mean_std_dev(y)

#     new_x = x[np.where(np.logical_and(x>=(x_mean-x_std), x<=(x_mean+x_std)))]
#     new_y = y[np.where(np.logical_and(y>=(y_mean-y_std), y<=(y_mean+y_std)))]

  
#     a = __fit_ellipse(new_x, new_y)
#     centre = ellipse_center(a)
#     phi = ellipse_angle_of_rotation(a)
#     M, m = ellipse_axis_length(a)
#     # assert that the major axix M > minor axis m
#     if m > M:
#         M, m = m, M
#     # ensure the angle is betwen 0 and 2*pi
#     phi -= 2 * np.pi * int(phi / (2 * np.pi))
#     print(M, m, centre[0], centre[1], phi)
#     return [M, m, centre[0], centre[1], phi]


def find_mean_std_dev(X):
    mean = 0
    std_dev = 0

    mean = np.mean(X)
    std_dev = np.std(X)

    print(mean,std_dev)

    lower = mean - std_dev
    upper = mean + std_dev

    # print(len(X),upper-lower,lower,upper,((lower<X)&(X<upper)).sum())
    # print("")
    return [lower, upper]


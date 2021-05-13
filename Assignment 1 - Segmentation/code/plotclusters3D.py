import cv2
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat("../data/pts.mat")
data = np.array(mat['data'])  # data set of 3D points
data = data.transpose()

# TODO: reshape data (with code below) and check code
# suggestions during the tutorial:
# points = loadmat("data/pts.mat")["data"].reshape(-1, 3)
# mpl_toolkits.mplot3d


def closest_node(node, nodes, d):
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.where(dist_2 < d)


def findpeak(data, idx, r, t=0.01):
    points_in_circle = np.empty([0, 3])
    # find first mean at index position
    for point in data:
        if math.sqrt(pow(point[0] - idx[0], 2) +
                     pow(point[1] - idx[1], 2) +
                     pow(point[2] - idx[2], 2)) < r:

            points_in_circle = np.vstack([points_in_circle, point])

    # points_in_circle = closest_node(point, data, r)
    mean = np.mean(points_in_circle, axis=0)

    # move towards peak
    search = True
    while search:
        new_points_in_circle = np.empty([0, 3])
        for point in data:
            if math.sqrt(pow(point[0] - mean[0], 2) +
                         pow(point[1] - mean[1], 2) +
                         pow(point[2] - mean[2], 2)) < r:
                new_points_in_circle = np.vstack([new_points_in_circle, point])
        # new_points_in_circle = closest_node(mean, data, r)
        new_mean = np.mean(new_points_in_circle, axis=0)
        if (mean[0] + t > new_mean[0] > mean[0] - t) \
                and (mean[1] + t > new_mean[1] > mean[1] - t) \
                and (mean[2] + t > new_mean[2] > mean[2] - t):
            search = False
        else:
            mean = new_mean

    return mean


def meanshift(data, r):
    labels = np.zeros(len(data))  # labels are numbers TODO 1d shape -> 2d shape
    peaks = np.empty([0, 3])

    label = 1
    label_idx = 0
    for point in data:
        print(point)
        new_peak = findpeak(data, point, r)
        peak_found = False
        for peak_idx, peak in enumerate(peaks):
            if math.sqrt(pow(peak[0] - new_peak[0], 2) +
                         pow(peak[1] - new_peak[1], 2) +
                         pow(peak[2] - new_peak[2], 2)) < r/2:
                labels[label_idx] = peak_idx
                label_idx +=1
                peak_found = True
                break
        if not peak_found:
            labels[label_idx] = label
            label_idx += 1
            label +=1
            peaks = np.vstack([peaks, new_peak])

    return labels, peaks


# TODO List with labeled and unlabeled datapoints
# first speedup: basin of attraction
def meanshift_opt (data, r, c=4):
    labels = np.zeros(len(data))  # labels are numbers
    peaks = np.empty([0, 3])

    label = 1
    label_idx = 0
    for point in data:
        if not labels[label_idx] == 0:
            pass
        new_peak = findpeak(data, point, r)
        peak_found = False
        for peak in range(len(peaks)):
            if math.sqrt(pow(peaks[peak][0] - new_peak[0], 2) +
                         pow(peaks[peak][1] - new_peak[1], 2) +
                         pow(peaks[peak][2] - new_peak[2], 2)) < r / 2:
                labels[label_idx] = peak
                label_idx += 1
                peak_found = True
                new_peak = peak
                break
        if not peak_found:
            labels[label_idx] = label
            label_idx += 1
            label += 1
            peaks = np.vstack([peaks, new_peak])

        toLabel = closest_node(new_peak, data, r)
        for point in toLabel:
             labels[point] = label

    return labels, peaks

# second speed up: points along path
def find_peak_opt(data, idx, r, threshold, c=4):

    pass


def plotclusters3D(data, labels, peaks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[...,::-1]
    rgb_peaks /= 255.0
    for idx, peak in enumerate(rgb_peaks):
        color = np.random.uniform(0, 1, 3)
        # TODO: instead of random color, you can use peaks when you work on actual images
        # color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    print("showing figure")
    fig.show()


r = 2  # should give two clusters
# labels, peaks = meanshift_opt(data, r)
# labels, peaks = meanshift(data, r)
# plotclusters3D(data, labels, peaks)
# TODO: experiments - measure runtime?


# TODO: image preprocessing
# resize, blur, RGB to LAB

img = cv2.imread('../data/img-1.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# img = img.transpose()
labels, peaks = meanshift(img, r)
plotclusters3D(img, labels, peaks)

# for point in img:
#     print(point)

# data = data.transpose()
# for point in data:
#     print(point)

plt.imshow(img)
plt.show()

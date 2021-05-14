import cv2
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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
    idx = idx.reshape(1,-1)

    # find first mean at index position
    distances = cdist(data, idx, metric='euclidean')
    indexes = np.where(distances < r)
    for index in indexes[0]:
        points_in_circle = np.vstack([points_in_circle, data[index]])
    mean = np.mean(points_in_circle, axis=0)

    # move towards peak
    search = True
    while search:
        new_points_in_circle = np.empty([0, 3])
        mean = mean.reshape(1,-1)

        distances = cdist(data, mean, metric='euclidean')
        indexes = np.where(distances < r)
        for index in indexes[0]:
            new_points_in_circle = np.vstack([new_points_in_circle, data[index]])
        new_mean = np.mean(new_points_in_circle, axis=0)

        new_mean = new_mean.reshape(1, -1)
        d = cdist(mean, new_mean, metric='euclidean')
        if d < t:
            search = False
        else:
            mean = new_mean

    return mean


def meanshift(data, r):
    labels = np.zeros(len(data))  # labels are numbers TODO 1d shape -> 2d shape
    peaks = np.empty([0, 3])

    label_peak = dict()
    label_peak[1] = 2

    label = 0  # has to be zero because of plot function
    label_idx = 0
    for i, point in enumerate(data):
        # print(point)
        new_peak = findpeak(data, point, r)
        peak_found = False
        distances = cdist(peaks, new_peak, metric='euclidean')
        distances = distances.flatten()
        indexes = np.where(distances < r/2)
        # indexes = indexes.flatten()
        # find peak in data and find label

        if indexes[0].size != 0:
            # a = np.where(data == peaks[0][indexes[0]])
            labels[i] = label_peak[str(peaks[indexes[0]])]
            label_idx +=1
        # for peak_idx, peak in enumerate(peaks):
        #     if math.sqrt(pow(peak[0] - new_peak[0], 2) +
        #                  pow(peak[1] - new_peak[1], 2) +
        #                  pow(peak[2] - new_peak[2], 2)) < r/2:
        #         labels[label_idx] = peak_idx #TODO check here
        #         label_idx +=1
        #         peak_found = True
        #         break
        else:
            print("here")
            print(i)
            label_peak[str(new_peak)] = label
            labels[i] = label
            label_idx += 1
            label +=1
            peaks = np.vstack([peaks, new_peak])

    return labels, peaks


# TODO List with labeled and unlabeled datapoints
# first speedup: basin of attraction
# TODO labels not working!!! -> commented code
def meanshift_opt (data, r, c=4):
    labels = np.zeros(len(data))  # labels are numbers
    peaks = np.empty([0, 3])

    label = 1
    label_idx = 0
    for point in data:
        # if not labels[label_idx] == 0:
        #     pass
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
labels, peaks = meanshift(data, r)
plotclusters3D(data, labels, peaks)
# TODO: experiments - measure runtime?


# TODO: image preprocessing
# resize, blur, RGB to LAB

img = cv2.imread('../data/img-1.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# img = img.transpose()
img = img.transpose(2,0,1).reshape(3,-1)
img = img.transpose()

# labels, peaks = meanshift(img, r)
# plotclusters3D(img, labels, peaks)

# for point in img:
#     print(point)

# data = data.transpose()
# for point in data:
#     print(point)

# plt.imshow(img)
# plt.show()

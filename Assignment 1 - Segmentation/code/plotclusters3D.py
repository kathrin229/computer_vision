import cv2
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage import color

mat = scipy.io.loadmat("../data/pts.mat")
data = np.array(mat['data'])  # data set of 3D points
data = data.transpose()

# suggestions during the tutorial:
# points = loadmat("data/pts.mat")["data"].reshape(-1, 3)
# mpl_toolkits.mplot3d

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
    labels = np.zeros(len(data))  # labels are numbers
    peaks = np.empty([0, 3])

    label_peak = dict()

    label = 0  # has to be zero because of plot function
    for i, point in enumerate(data):
        new_peak = findpeak(data, point, r)
        distances = cdist(peaks, new_peak, metric='euclidean')
        indexes = np.where(distances < r/2)

        # close peak already exists: same label
        if indexes[0].size != 0:
            print(i)
            labels[i] = label_peak['['+str(peaks[indexes[0][0]])+']']

        else:
            label_peak[str(new_peak)] = label
            labels[i] = label
            label +=1
            peaks = np.vstack([peaks, new_peak])

    return labels, peaks


# TODO List with labeled and unlabeled datapoints
# first speedup: basin of attraction
# TODO labels not working!!! -> commented code
def meanshift_opt (data, r, c=4):
    labels = np.empty(len(data))  # labels are numbers #TODO: fill with -1
    labels.fill(-1)
    peaks = np.empty([0, 3])
    print(labels)

    label_peak = dict()

    label = 0  # has to be zero because of plot function
    #for i, point in enumerate(data): # TODO: eliminate for loop here
    while True:
        checkCondition = np.where(labels == -1)[0]
        # print("size", i[0].size)
        if(checkCondition.size) == 0:
            print("here")
            break
        i = checkCondition[0]
        point = data[i]
        new_peak, label_points = find_peak_opt(data, point, r)
        distances = cdist(peaks, new_peak, metric='euclidean')
        indexes = np.where(distances < r / 2)

        # close peak already exists: same label
        if indexes[0].size != 0:
            print("1")
            print(i)
            labels[i] = label_peak['[' + str(peaks[indexes[0][0]]) + ']']
            new_peak = peaks[indexes[0][0]]
            new_peak = new_peak.reshape(1, -1)
            distances = cdist(data, new_peak, metric='euclidean')
            index_close_points = np.where(distances < r)
            new_label = label_peak['[' + str(peaks[indexes[0][0]]) + ']']
            for x in index_close_points[0]:
                labels[x] = new_label
            for label_point in label_points:
                labels[int(label_point)] = new_label

        else:
            print("2")
            label_peak[str(new_peak)] = label
            labels[i] = label
            peaks = np.vstack([peaks, new_peak])
            new_peak = new_peak.reshape(1, -1)
            distances = cdist(data, new_peak, metric='euclidean')
            index_close_points = np.where(distances < r)
            for x in index_close_points[0]:
                labels[x] = label
            for label_point in label_points:
                labels[int(label_point)] = label
            label += 1

    return labels, peaks


# second speed up: points along path
def find_peak_opt(data, idx, r, t = 0.01, c=4):
    points_in_circle = np.empty([0, 3])
    idx = idx.reshape(1, -1)

    peak_indexes = np.empty([0, 1])

    # find first mean at index position
    distances = cdist(data, idx, metric='euclidean')
    indexes = np.where(distances < r)
    for index in indexes[0]:
        points_in_circle = np.vstack([points_in_circle, data[index]])
    mean = np.mean(points_in_circle, axis=0)

    # TODO: for every mean, find surrounding points
    # array with points that have to be labeled
    # peak_points = np.empty([0, 3])

    # move towards peak
    search = True
    while search:
        new_points_in_circle = np.empty([0, 3])
        mean = mean.reshape(1, -1)

        distances = cdist(data, mean, metric='euclidean')
        indexes = np.where(distances < r)
        for index in indexes[0]:
            new_points_in_circle = np.vstack([new_points_in_circle, data[index]])
        new_mean = np.mean(new_points_in_circle, axis=0)
        new_mean = new_mean.reshape(1, -1)

        a = cdist(data, new_mean, metric='euclidean')
        points = np.where(a < c)[0]
        peak_indexes = np.append(peak_indexes, points)

        # for point in points[0]:
        #     peak_points = np.vstack([peak_points, data[point]])

        d = cdist(mean, new_mean, metric='euclidean')
        if d < t:
            search = False
        else:
            mean = new_mean
    # return mean (peak) and points that should be labeled
    return mean, peak_indexes
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
        color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    print("showing figure")
    fig.show()


# def segmIm(im, r):


r = 40  # should give two clusters
# labels, peaks = meanshift_opt(data, r)
# labels, peaks = meanshift(data, r) # WORKS!
# plotclusters3D(data, labels, peaks)
# TODO: experiments - measure runtime?


# TODO: image preprocessing
# resize, blur, RGB to LAB

img = cv2.imread('../data/img-3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
print(img[0][0])
plt.imshow(img)
plt.show()
img_resize = cv2.resize(img, (30, 40), interpolation=cv2.INTER_NEAREST)
img = cv2.resize(img, (30, 40), interpolation=cv2.INTER_NEAREST)
plt.imshow(img)
plt.show()
img_height = 40 #len(img_resize[0])
img_width = 30 #len(img_resize[1])
img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img_resize = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# img = color.rgb2lab(img)
plt.imshow(img)
plt.show()

img = img.transpose(2,0,1).reshape(3,-1)
img = img.transpose()

labels, peaks = meanshift_opt(img, r)
# peaks = peaks.reshape(len(peaks), 3, 1).transpose(0,2,1)
# peaks = cv2.cvtColor(peaks, cv2.COLOR_LAB2RGB) TODO maybe ints solve problem
# plotclusters3D(img, labels, peaks)

# for idx in range(len(img)):
#     img[idx] = peaks[int(labels[idx])]
x = 0
for i in range(img_height):
    for j in range(img_width):
        img_resize[i][j] = peaks[int(labels[x])]
        x += 1

# img = img.reshape(3, img_height, img_width).transpose(1, 2, 0)
img_resize = cv2.cvtColor(img_resize, cv2.COLOR_LAB2RGB)
# img_resize = color.lab2rgb(img_resize)
plt.imshow(img_resize)
plt.show()



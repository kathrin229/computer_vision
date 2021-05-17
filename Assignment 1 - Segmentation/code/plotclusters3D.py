import cv2
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from skimage import color


mat = scipy.io.loadmat("../data/pts.mat")
data = np.array(mat['data'])  # data set of 3D points
data = data.transpose()

# suggestions during the tutorial:
# points = loadmat("data/pts.mat")["data"].reshape(-1, 3)
# mpl_toolkits.mplot3d

def findpeak(data, idx, r, t=0.01):
    idx = idx.reshape(1,-1)

    # find first mean at index position
    distances = cdist(data, idx, metric='euclidean')
    indexes = np.where(distances < r)[0]
    points_in_circle = data[indexes]
    mean = np.mean(points_in_circle, axis=0)

    # move towards peak
    search = True
    while search:
        mean = mean.reshape(1,-1)

        distances = cdist(data, mean, metric='euclidean')
        indexes = np.where(distances < r)[0]
        new_points_in_circle = data[indexes]
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

        # if close peak already exists: same label
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
def meanshift_opt (data, r, c=4, feature_space='3D'):
    labels = np.empty(len(data))  # labels are numbers, filled with -1
    labels.fill(-1)
    if feature_space == '3D':
        peaks = np.empty([0, 3])
    if feature_space == '5D':
        peaks = np.empty([0, 5])

    label_peak = dict()

    label = 0  # has to be zero because of plot function
    while True:
        points_without_label = np.where(labels == -1)[0]
        if points_without_label.size == 0:
            print("here")
            break
        i = points_without_label[0]
        if i % 10 == 0:
            print(i)
        point = data[i]
        new_peak, label_points = find_peak_opt(data, point, r, c)
        distances = cdist(peaks, new_peak, metric='euclidean')
        indexes = np.where(distances < r / 2)

        # close peak already exists: same label
        if indexes[0].size != 0:
            labels[i] = label_peak['[' + str(peaks[indexes[0][0]]) + ']']
            new_peak = peaks[indexes[0][0]]
            new_peak = new_peak.reshape(1, -1)

            distances = cdist(data, new_peak, metric='euclidean')
            index_close_points = np.where(distances < r)[0]
            new_label = label_peak['[' + str(peaks[indexes[0][0]]) + ']']
            labels[index_close_points] = new_label
            labels[label_points. astype(int)] = new_label

        else:
            label_peak[str(new_peak)] = label
            labels[i] = label
            peaks = np.vstack([peaks, new_peak])
            new_peak = new_peak.reshape(1, -1)

            distances = cdist(data, new_peak, metric='euclidean')
            index_close_points = np.where(distances < r)[0]
            labels[index_close_points] = label
            labels[label_points. astype(int)] = label
            label += 1

    return labels, peaks


# second speed up: points along path
def find_peak_opt(data, idx, r, c, t = 0.01):
    idx = idx.reshape(1, -1)
    peak_indexes = np.empty([0, 1])

    # find first mean at index position
    distances = cdist(data, idx, metric='euclidean')
    indexes = np.where(distances < r)[0]
    points_in_circle = data[indexes]
    mean = np.mean(points_in_circle, axis=0)
    # TODO: find surrounding points for first mean


    # move towards peak
    search = True
    while search:
        mean = mean.reshape(1, -1)

        distances = cdist(data, mean, metric='euclidean')
        indexes = np.where(distances < r)[0]
        new_points_in_circle = data[indexes]
        new_mean = np.mean(new_points_in_circle, axis=0)
        new_mean = new_mean.reshape(1, -1)

        a = cdist(data, new_mean, metric='euclidean')
        points = np.where(a < c)[0]
        peak_indexes = np.append(peak_indexes, points)

        d = cdist(mean, new_mean, metric='euclidean')
        if d < t:
            search = False
        else:
            mean = new_mean
    # return mean (peak) and points that should be labeled
    return mean, peak_indexes
    pass


def plotclusters3D(data, labels, rgb_peaks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    # rgb_peaks = bgr_peaks[...,::-1]
    rgb_peaks = rgb_peaks.astype(float)
    rgb_peaks /= 255
    for idx, peak in enumerate(rgb_peaks):
        color = np.random.uniform(0, 1, 3)
        # TODO: instead of random color, you can use peaks when you work on actual images
        color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    print("showing figure")
    plt.savefig('Space')
    fig.show()



def segmIm(im, r, c, feature_space='3D'):
    start_time = time.time()
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_width, img_height, _ = img.shape

    # TODO: image preprocessing
    # resize, blur, RGB to LAB
    img = cv2.resize(img, (int(img_height/2), int(img_width/2)), interpolation=cv2.INTER_NEAREST)

    if feature_space == '5D':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        yx_coords = np.column_stack(np.where(gray >= 0))

    img_original = img

    # kernel = np.ones((5, 5), np.float32) / 25
    # img = cv2.filter2D(img, -1, kernel)

    img_width, img_height, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img = img.transpose(2, 0, 1).reshape(3, -1)
    img = img.transpose()

    if feature_space == '5D':
        img = np.append(img, yx_coords, axis=1)

    labels, peaks = meanshift_opt(img, r, c, feature_space)

    if feature_space == '5D':
        peaks = np.delete(peaks, 3, 1)
        peaks = np.delete(peaks, 3, 1)

    plt.imshow(img_original)
    plt.show()

    x = 0
    for i in range(img_width):
        for j in range(img_height):
            img_original[i][j] = peaks[int(labels[x])]
            x += 1

    img_resize = cv2.cvtColor(img_original, cv2.COLOR_LAB2RGB)

    peaks = img_resize.transpose(2, 0, 1).reshape(3, -1)
    peaks = peaks.transpose()

    peaks = np.unique(peaks, axis=0)


    print("--- %s seconds ---" % (int(time.time() - start_time)))

    plt.imshow(img_original)
    plt.savefig('Lab')
    plt.show()


    plt.imshow(img_resize)
    plt.savefig('Segment')
    plt.show()
    plotclusters3D(img, labels, peaks)



r = 30  # 2 should give two clusters
c = 10
feature_space = '5D'

start_time = time.time()
labels, peaks = meanshift_opt(data, r)
# labels, peaks = meanshift(data, r) # WORKS!
plotclusters3D(data, labels, peaks)
# TODO: experiments - measure runtime?
print("--- %s seconds ---" % (time.time() - start_time))

load_img = cv2.imread('../data/img-3.jpg')
# segmIm(load_img, r, c,  feature_space=feature_space)

# speedup ideas
# - where different usage
# - list conversion to np array by np concatenate? instead of vstack

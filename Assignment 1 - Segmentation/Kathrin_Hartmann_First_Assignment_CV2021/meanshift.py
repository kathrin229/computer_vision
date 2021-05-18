import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist


def main():
    # Loading data and image
    mat = scipy.io.loadmat("../data/pts.mat")
    data = np.array(mat['data'])
    data = data.transpose()

    load_img = cv2.imread('../data/img-3.jpg')

    # Parameters of the algorithm
    r = 10
    c = 30
    feature_space = '5D'  # '3D' or '5D'

    # Testing algorithm on dataset pts.mat
    # labels, peaks = meanshift_opt(data, r)
    # labels, peaks = meanshift(data, r) # WORKS!
    # plotclusters3D(data, labels, peaks)

    # Image segmentation
    segmIm(load_img, r, c,  feature_space=feature_space)


def findpeak(data, idx, r, t=0.01):
    '''

    Args:
        data: The data points in an channel x (X x Y) dimensional array
        idx: the index for which the peak has to be found
        r: the size of the basin of attraction
        t: parameter avoiding too close peaks

    Returns: the final mean (the peak)

    '''
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
    '''

        Args:
            data: The data points in an channel x (X x Y) dimensional array
            r: the size of the basin of attraction

        Returns: an array of labels and an array with corresponding peaks

        '''
    labels = np.zeros(len(data))
    peaks = np.empty([0, 3])

    label_peak = dict()

    label = 0
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


# first speedup: basin of attraction
def meanshift_opt (data, r, c=4, feature_space='3D'):
    '''

    Args:
        data: The data points in an channel x (X x Y) dimensional array
        r: the size of the basin of attraction
        c: parameter for the second speedup - how many points "on the way to the peak" belong to the peak
        feature_space: 3D of 5D feature space

    Returns: an array of labels and an array with corresponding peaks

    '''
    labels = np.empty(len(data))  # labels are numbers, filled with -1 initially
    labels.fill(-1)
    if feature_space == '3D':
        peaks = np.empty([0, 3])
    if feature_space == '5D':
        peaks = np.empty([0, 5])

    label_peak = dict()

    label = 0
    while True:
        points_without_label = np.where(labels == -1)[0]
        if points_without_label.size == 0:
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


def find_peak_opt(data, idx, r, c, t=0.01):
    '''
    second speed up
    Args:
        data: The data points in an channel x (X x Y) dimensional array
        idx: the index for which the peak has to be found
        r: the size of the basin of attraction
        c: parameter for the second speedup - how many points "on the way to the peak" belong to the peak
        t: parameter avoiding too close peaks

    Returns: the final mean (the peak) and points found on the way that belong to this peak

    '''
    idx = idx.reshape(1, -1)
    peak_indexes = np.empty([0, 1])

    # find first mean at index position
    distances = cdist(data, idx, metric='euclidean')
    indexes = np.where(distances < r)[0]
    points_in_circle = data[indexes]
    mean = np.mean(points_in_circle, axis=0)

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


def plotclusters3D(data, labels, rgb_peaks):
    '''

    Args:
        data: The data points
        labels: Array of labels for each data point
        rgb_peaks: Existing peaks

    Returns: no return

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # bgr_peaks = np.array(peaks[:, 0:3], dtype=float)  # commented out for image segmentation
    # rgb_peaks = bgr_peaks[...,::-1]  # commented out for image segmentation
    rgb_peaks = rgb_peaks.astype(float)
    rgb_peaks /= 255
    for idx, peak in enumerate(rgb_peaks):
        # color = np.random.uniform(0, 1, 3) # commented out for image segmentation
        color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    print("showing figure")
    fig.show()


def segmIm(im, r, c, feature_space='3D'):
    '''

    Args:
        im: The image read by cv2
        r: The size of the basin of attraction
        c: Parameter for the second speedup
        feature_space: Either '3D' or '5D' for the different feature spaces

    Returns: no returns. Plots of original image, image in LAB colour space, segmented image and clusters

    '''
    # Timer
    start_time = time.time()

    # Image preprocessing:
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_width, img_height, _ = img.shape
    img = cv2.resize(img, (int(img_height/2), int(img_width/2)), interpolation=cv2.INTER_NEAREST)

    if feature_space == '5D':
        coord_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        yx_coords = np.column_stack(np.where(coord_image >= 0))

    img_original = img

    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)

    img_width, img_height, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img = img.transpose(2, 0, 1).reshape(3, -1)
    img = img.transpose()

    if feature_space == '5D':
        img = np.append(img, yx_coords, axis=1)

    # Calling the mean-shift algorithm:
    labels, peaks = meanshift_opt(img, r, c, feature_space)

    # Creation of segmented image:
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

    img_cluster = cv2.cvtColor(img_original, cv2.COLOR_LAB2RGB)

    peaks = img_cluster.transpose(2, 0, 1).reshape(3, -1)
    peaks = peaks.transpose()
    peaks = np.unique(peaks, axis=0)

    print("--- %s seconds ---" % (int(time.time() - start_time)))

    plt.imshow(img_original)
    plt.show()

    plt.imshow(img_cluster)
    plt.show()
    plotclusters3D(img, labels, peaks)



if __name__ == "__main__":
    main()

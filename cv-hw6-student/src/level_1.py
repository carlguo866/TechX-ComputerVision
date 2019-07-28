"""
TechX Academy CV Homework 6
Lecturer: Cecilia Zhang
Homework created by Yijia Chen, Ruijie Fang
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import optional
"""
Helper utilities 
"""


def read_house_image():
    """
    Reads two images of a house object, seen from different angles
    :return: a 2-tuple (house1, house2), each of which is a cv2 grayscale
    image
    """
    house1 = cv2.imread('../data/house1.png', cv2.IMREAD_GRAYSCALE)
    house2 = cv2.imread('../data/house2.png', cv2.IMREAD_GRAYSCALE)
    return house1, house2


def read_library_image():
    """
    Reads two images of a library object,  seen from two different angles
    :return: a 2-tuple (house1, house2), each of which is a cv2 grayscale
    """
    library1 = cv2.imread('../data/library1.png', cv2.IMREAD_GRAYSCALE)
    library2 = cv2.imread('../data/library2.png', cv2.IMREAD_GRAYSCALE)
    return library1, library2


def plot_images_side_by_side(image1, image2):
    """
    Plots two cv2 images side by side using pyplot.
    :param image1: the image on the left
    :param image2: the image on the right
    :return: NoneType
    """
    figure = plt.figure()
    figure.add_subplot(1, 2, 1)
    plt.imshow(image1)
    figure.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()


def plot_single_image(image):
    """
    Plots a single image using pyplot.
    :param image: The image to be plotted.
    :return: NoneType
    """
    plt.imshow(image)
    plt.show()


"""
Feature detection and matching
"""


class ORBDetectorResult(object):
    """
    Class that stores the result of detect_feature_points
    """

    def __init__(self, keypoints1, keypoints2, descriptor1, descriptor2,
                 orb_matches):
        """
        Initializes an ORBDetectorResult instance
        :param keypoints1: keypoints list from first image
        :param keypoints2: keypoints list from second image
        :param descriptor1: descriptors list from first image
        :param descriptor2: descriptors list from second image
        :param orb_matches: sorted list (by hamming dist) of matched points
        """
        self.keypoints1 = keypoints1
        self.keypoints2 = keypoints2
        self.descriptor1 = descriptor1
        self.descriptor2 = descriptor2
        self.orb_matches = orb_matches


def detect_feature_points(image1, image2):
    """
    Uses an orb detector to detect and match feature points between two images.
    :param image1: Image of an object
    :param image2: Another image of the same object
    :return: an ORBDetectorResult object
    """

    orb_detector = cv2.ORB_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    keypoints1, descriptor1 = orb_detector.detectAndCompute(image1, None)
    keypoints2, descriptor2 = orb_detector.detectAndCompute(image2, None)
    orb_matches = sorted(bf_matcher.match(descriptor1, descriptor2),
                         key=lambda x: x.distance)
    return ORBDetectorResult(keypoints1, keypoints2, descriptor1, descriptor2,
                             orb_matches)


"""
Fundamental matrix calculation and and image warping
"""


def print_fundamental_matrix(points1, points2, orb_detection_result):
    """
    Prints the fundamental matrix to command line
    :param image1: the first image
    :param image2: the second image
    :param orb_detection_result: of type ORBDetectorResult
    :return: NoneType
    """
    # type safety check
    assert (str(type(orb_detection_result))[-19:].strip('>').strip('\'')
            == 'ORBDetectorResult')

    """
    PROBLEM 1: Compute the fundamental matrix using RANSAC.
    """
    #################### YOUR CODE HERE ####################
    mat, mask = cv2.findFundamentalMat(points1,points2)
    ########################################################

    print('*-*-*-*-*-*-*-*-* Fundamental Matrix *-*-*-*-*-*-*-*-*')
    print(mat)
    print('*-*-*-*-*-*-*-*-* Fundamental Matrix *-*-*-*-*-*-*-*-*')


def calculate_alignment(image1, image2, orb_detection_result, num_good_matches=15):
    """
    Calculates an image alignment based on orb detection result.
    To make sure RANSAC returns the _best_ result, your # matches must >= 8.
    Empirical evidence suggests that you might want to try including more matched points
    rather than less.
    :param image1: the first image
    :param image2: the second image
    :param orb_detection_result: of ORBDetectorResult type,
    as returned by detect_feature_points
    :param num_good_matches: the total number of good matches (from the start
    of the ORBDetectorResult::orb_matches list) to take
    :return: a 2-tuple (h, mask) to be fed into align_images(...)
    """
    # type safety check
    assert (str(type(orb_detection_result))[-19:].strip('>').strip('\'')
            == 'ORBDetectorResult')
    good_matches = orb_detection_result.orb_matches[:num_good_matches]

    # now, find locations of good_matches

    """
    PROBLEM 0: Create two arrays of matched points using orb_detection_result.
    """
    #################### YOUR CODE HERE ####################
    match_points_1 = []
    match_points_2 = []
    for match in good_matches:
        match_points_1.append(orb_detection_result.keypoints1[match.queryIdx].pt)
        match_points_2.append(orb_detection_result.keypoints2[match.trainIdx].pt)
    # im_match = cv2.drawMatches(image1, orb_detection_result.keypoints1, image2,
    #                            orb_detection_result.keypoints2, good_matches, None)
    # plt.figure(figsize=(15, 8))
    # plt.imshow(im_match)
    # plt.show()
    ########################################################
    match_points_1 = np.array(match_points_1)
    match_points_2 = np.array(match_points_2)
    # print(match_points_1)
    # print(match_points_2)
    # ** IMPORTANT **: Now, print the fundamental matrix, feed in the two matched points list
    print_fundamental_matrix(match_points_1, match_points_2, orb_detection_result)

    """
    PROBLEM 2: Calculate homography using RANSAC.
    """
    #################### YOUR CODE HERE ####################
    h, mask = cv2.findHomography(match_points_1, match_points_2)
    ########################################################

    return h, mask


def align_images(image1, image2, h):
    """
    Aligns the perspective of image 1 to image 2
     using (h, mask) calculated by calculate_alignment
    :param image1:
    :param image2:
    :param h:
    :param mask:
    :return:
    """
    # Use homography
    height = image2.shape[0]
    width = image2.shape[1]
    image1_reg = cv2.warpPerspective(image1, h, (width, height))

    """
    PROBLEM 3: Warp image1 using given h and dimensions of image2.
    """
    #################### YOUR CODE HERE ####################

    ########################################################

    return image1_reg, h


"""
Driver code
"""

house1, house2 = read_library_image()
matched = detect_feature_points(house1, house2)
# now, print first 10 matches
print(matched)

matched_image = cv2.drawMatches(house1, matched.keypoints1, house2,
                                matched.keypoints2,
                                matched.orb_matches[:15], None, flags=2)

plot_single_image(matched_image)

alignment = calculate_alignment(house1,
                                house2, matched, 30)
house1p, h = align_images(house1, house2, alignment[0])
plot_images_side_by_side(house1, house1p)

i,j = optional.preprocess_crop_black(house1p)
if i[0] < j[0]:
    minx = i[0]
    maxx = j[0]
else:
    minx = j[0]
    maxx = i[0]
if i[1] < j[1]:
    miny = i[1]
    maxy = j[1]
else:
    miny = j[1]
    maxy = i[1]
cropped_img = house1p[miny:maxy,minx:maxx ]
print(cropped_img.shape)
plt.imshow(cropped_img)
plt.show()
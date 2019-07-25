import numpy as np
import random
"""
Compute H.

Inputs:
p1, p2: N × 2 matrices specifying corresponding point locations by two cameras

Output:
H: 3 × 3 homography matrix that best matches the linear equations
"""


def compute_H(p1, p2):
    N = len(p1)
    print(N)
    A = np.zeros([N * 2,9])
    for i in range(np.int(N)):
        A[i * 2,:2] = p1[i]
        A[i * 2,2] = 1
        A[i * 2 + 1, 3:5] = p1[i]
        A[i * 2 + 1, 5] = 1
        A[i * 2, 6:8] = -p1[i] * p2[i, 0]
        A[i * 2, 8] = -1 * p2[i, 0]
        A[i * 2 + 1,6:8] = -p1[i] * p2[i, 1]
        A[i * 2 + 1,8] = -1 * p2[i, 1]
    c1, c2 = np.linalg.eig(A.T@ A)
    c2 = c2[np.argmin(c1)]
    H = np.reshape(c2, (3, 3))
    print(H)
    return H



"""
Compute homographies automatically between two images using RANSAC (Random Sample Convention) algorithm.

Function Inputs:
locs1, locs2: N × 2 matrices specifying point locations in each of the images
matches: N × 2 matrix specifying matches between locs1 and locs2

Algorithm Inputs:
n_iter: the number of iterations to run RANSAC for
tol: the tolerance value for considering a point to be an inlier

Output:
bestH: the homography model with the most inliers found during RANSAC

PARTIAL CREDITS:
- Find model for randomly selected points (10 pts)
- Extend model to all inliers of model (15 pts)
- Iterate correctly to get best-fitting H (15 pts)

Hint:
1. You can use "float('inf')" for a really big number.
2. You can use the given "match" function to get p1 and p2.
3. The model here is the compute_H you implement above.
"""


def ransac_H(matches, locs1, locs2, n_iter, tol):
    N = len(matches)  # length of locs1, locs2, and matches
    iter = 0
    fit_threshold = 3  # must be between 0 and N. Feel free to adjust.
    best_fit = 0
    bestErr = float('inf')
    while iter < n_iter:
        thisErr = 0

        indexes = random.sample(range(0, N), 4)
        maybeInliers = match(locs1, locs2, matches[indexes])
        maybeOutlierMatches = np.delete(matches, indexes, axis=0)
        maybeOutliers = match(locs1, locs2, maybeOutlierMatches)
        maybeModel = compute_H(maybeInliers[0], maybeInliers[1])
        alsoInliers = []

        for points in maybeOutliers:
            maybeErr = compute_error(points[0], points[1], maybeModel)
            if maybeErr < tol:
                thisErr += maybeErr
                alsoInliers.append(points)

        if alsoInliers[0].shape[0] > fit_threshold:
            betterModel = maybeModel

            for maybeInlier in maybeInliers:
                thisErr += compute_error(maybeInlier[0], maybeInlier[1], betterModel)

            if thisErr < bestErr:
                bestFit = betterModel
                bestErr = thisErr

        iter+=1
    #print(best_err)
    return best_fit

def match(locs1, locs2, pairs):
    N = len(pairs)
    p1 = p2 = np.zeros((N, 2))
    for i in range(len(pairs)):
        p1[i] = locs1[pairs[i][0]]
        p2[i] = locs2[pairs[i][1]]
    return (p1, p2)


def compute_error(loc1, loc2, H):
    return np.linalg.norm(np.matmul(H, np.append(loc2, 1)) - np.append(loc1, 1))  # TODO: Find this way?


p1 = np.array([[382, 79], [395, 98], [419, 119], [448, 107],
               [388, 132], [410, 154], [383, 193], [403, 182], [408, 210], [450, 264],
               [483, 180], [488, 212], [581, 261], [514, 109], [530, 126], [475, 172],
               [519, 194], [489, 320], [555, 311], [543, 189], [579, 104], [603, 95],
               [552, 151], [580, 201], [523, 213], [554, 277], [600, 211], [385, 252],
               [442, 179], [469, 52]])
p2 = np.array([[31, 60], [40, 80], [71, 110], [99, 106], [38, 121],
               [56, 147], [21, 187], [47, 177], [41, 208], [86, 260], [127, 183],
               [129, 213], [216, 263], [166, 119], [180, 134], [128, 176], [166, 199],
               [127, 324], [188, 310], [190, 195], [221, 126], [240, 121], [198, 162],
               [218, 209], [166, 219], [189, 279], [233, 216], [20, 256], [92, 175],
               [125, 52]])
matches = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10],
                    [11, 11], [12, 12], [13, 13], [14, 14], [15, 0], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20],
                    [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [26, 26], [27, 27], [28, 28], [29, 29], [0, 15]])
# print(p1)

H = compute_H(p1, p2)
best_H = ransac_H(matches, p1, p2, 30, 500)


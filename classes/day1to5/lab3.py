import numpy as np
import matplotlib.pyplot as plt
import math

# convert points from euclidian to homogeneous
def to_homog(points):
    """
    your code here
    """
    points_homog = np.zeros([points.shape[0]+1,points.shape[1]])
    points_homog[:-1,:] = points
    points_homog[-1,:] = 1
    #print(points_homog)
    return points_homog


# convert points from homogeneous to euclidian
def from_homog(points_homog):
    """
    your code here
    """
    print(points_homog[-1,:])
    points = points_homog[:-1, :]
    for i in range(points_homog.shape[1]):
        points[:,i]/=points_homog[-1,i]
    # print(points)
    return points


# project 3D euclidian points to 2D euclidian
def project_points(P_int, P_ext, pts):
    """
    your code here
    """

    # return the 2d euclidiean points
    ext = np.matmul(P_ext,to_homog(pts))
    # print(np.matmul(P_int,ext))
    pts_2d = from_homog(np.matmul(P_int,ext))
    return pts_2d



def camera1():
    """
    replace with your code
    """
    P_int_proj = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0]])
    P_ext = np.eye(4, 4)
    return P_int_proj, P_ext


def camera2():
    """
    replace with your code
    """
    P_int_proj = np.eye(3, 4)
    P_ext = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
    return P_int_proj, P_ext


def camera3():
    """
    replace with your code
    """
    P_int_proj = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
    sinz = math.sin(math.radians(30))
    cosz = math.cos(math.radians(30))
    siny = math.sin(math.radians(60))
    cosy = math.cos(math.radians(60))
    rotated_z = np.array([[cosz,-sinz,0,0],
                          [sinz,cosz,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])
    rotated_y = np.array([[cosy, 0, siny, 0],
                          [0, 1, 0, 0],
                          [-siny, 0, cosy, 0],
                          [0, 0, 0, 1]])
    P_ext = np.matmul(rotated_z,rotated_y)
   # print(P_ext[2,3])
    P_ext[2,3] = 1
  #  print(P_ext)
    return P_int_proj, P_ext


def camera4():
    """
    replace with your code
    """
    P_int_proj = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
    sinz = math.sin(math.radians(30))
    cosz = math.cos(math.radians(30))
    siny = math.sin(math.radians(60))
    cosy = math.cos(math.radians(60))
    rotated_z = np.array([[cosz, -sinz, 0, 0],
                          [sinz, cosz, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    rotated_y = np.array([[cosy, 0, siny, 0],
                          [0, 1, 0, 0],
                          [-siny, 0, cosy, 0],
                          [0, 0, 0, 1]])
    P_ext = np.matmul(rotated_z, rotated_y)
    print(P_ext[2, 3])
    P_ext[2][3] = 13
    print(P_ext)
    return P_int_proj, P_ext


# test code. Do not modify

def plot_points(points, title='', style='.-r', axis=[]):
    inds = list(range(points.shape[1])) + [0]
    plt.plot(points[0, inds], points[1, inds], style)
    if title:
        plt.title(title)
    if axis:
        plt.axis('scaled')
        # plt.axis(axis)


def main():
    point1 = np.array([[-1, -0.5, 2]]).T
    point2 = np.array([[1, -0.5, 2]]).T
    point3 = np.array([[1, 0.5, 2]]).T
    point4 = np.array([[-1, 0.5, 2]]).T
    points = np.hstack((point1, point2, point3, point4))


    for i, camera in enumerate([camera1, camera2, camera3, camera4]):
        P_int_proj, P_ext = camera()
        plt.subplot(2, 2, i+1)
        plot_points(project_points(P_int_proj, P_ext, points), title='Camera %d Projective' % (i+ 1),
                    axis=[-0.6, 0.6, -0.6, 0.6])
    plt.savefig('lab3.png')
    plt.show()

main()

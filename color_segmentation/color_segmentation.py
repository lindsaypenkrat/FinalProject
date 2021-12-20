import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

rat = cv2.imread('PATH_TO_IMAGE') #put path to rat.png here
rat = cv2.cvtColor(rat, cv2.COLOR_BGR2RGB)
hsv_rat = cv2.cvtColor(rat, cv2.COLOR_RGB2HSV)

def show_rat():
    plt.imshow(rat)
    plt.show()

def rgb_plot():
    r, g, b = cv2.split(rat) #splits image into component channels
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = rat.reshape((np.shape(rat)[0]*np.shape(rat)[1], 3)) 
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist() #flatten colors corrresponding to each pixel into a list
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()


def hsv_plot(): #HSV is more effective for localizing the pixels based on hue 
    h, s, v = cv2.split(hsv_rat)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = rat.reshape((np.shape(rat)[0]*np.shape(rat)[1], 3)) 
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def mask():

    light_white = (-.01,.03, 0)
    dark_white = (-.01, .03, 130)
    mask_white = cv2.inRange(hsv_rat, light_white, dark_white)
    result_white = cv2.bitwise_and(rat, rat, mask=mask_white)
    plt.subplot(1, 2, 1)
    plt.imshow(mask_white, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result_white)
    plt.show()
    final_mask = mask_white
  
    final_result = cv2.bitwise_and(rat, rat, mask=final_mask)
    plt.subplot(1, 2, 1)
    plt.imshow(final_mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(final_result)
    plt.show()

    return final_result
mask()

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb


 
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
bgr2rgb = flags[48] #flag to convert from bgr to rgb color space

nemo = cv2.imread('nemo0.jpg')
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

def show_nemo():
    plt.imshow(nemo)
    plt.show()

def rgb_plot():
    r, g, b = cv2.split(nemo) #splits image into component channels
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3)) 
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist() #flatten colors corrresponding to each pixel into a list
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

def hsv_plot(): #HSV is more effective for localizing the pixels based on hue 
    h, s, v = cv2.split(hsv_nemo)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3)) 
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def mask():
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)
    lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
    do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0
    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(do_square))
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_to_rgb(lo_square))
    plt.show() #show lower and upper bounds of color range
    mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
    result = cv2.bitwise_and(nemo, nemo, mask=mask)
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)
    mask_white = cv2.inRange(hsv_nemo, light_white, dark_white)
    result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)
    plt.subplot(1, 2, 1)
    plt.imshow(mask_white, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result_white)
    plt.show()
    final_mask = mask + mask_white
    
    final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
    plt.subplot(1, 2, 1)
    plt.imshow(final_mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(final_result)
    plt.show()

    return final_result

    
def blur():
    final_result = mask()
    blur = cv2.GaussianBlur(final_result, (7, 7), 0)
    plt.imshow(blur)
    plt.show()

blur()
    
    




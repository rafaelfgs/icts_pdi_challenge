### Desafio Pratico de PDI - Grupo ICTS
### Rafael Fernandes Goncalves da Silva
### 01/08/2020 - Belo Horizonte

### Input and output variables in __main__
### function at the end of this script



### Required modules

# numpy (for array manipulation)
import numpy as np

# OpenCV(cv2) (for image manipulation)
import cv2

# ImUtils (for image rotation)
import imutils

# Math (for math functions)
from math import atan2, cos, sin, pi, sqrt



### Function to read the input file and convert it to HSV space
### Inputs: input file
### Outputs: HSV image

def read_hsv(input_file):
    
    # Read RGB image
    rgb_full = cv2.imread(input_file)
    
    # Resize the image to a 720p resolution
    rgb_size = rgb_full.shape
    rgb_res = 720
    rgb_width = int(round(float(rgb_res) * rgb_size[1] / rgb_size[0]))
    rgb = cv2.resize(rgb_full, (rgb_width, rgb_res))
    
    # Convert to HSV space
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    
    return hsv



### Function to segment the green color of the HSV image
### Inputs: HSV image
### Outputs: segmented image, segmented image size

def green_segmentation(hsv):
    
    # Image size
    img_size = hsv.shape
    
    # Set color to segmentation - 0:red, 30:yellow, 60:green, 90:cyan, 120:blue, 150:magenta
    hsv_color = 60
    
    # Threshold for segmentation - [range(Hue), min(Saturation), min(Value)]
    hsv_k = [18, 159, 95]
    
    # Segmentation of the selected color
    img_seg = ((hsv[:,:,0] > (hsv_color-hsv_k[0])) & 
               (hsv[:,:,0] < (hsv_color+hsv_k[0])) & 
               (hsv[:,:,1] > hsv_k[1]) & 
               (hsv[:,:,2] > hsv_k[2]))

    return img_seg, img_size



### Function to detect the top edge of the segmented image
### Inputs: segmented image, segmented image size
### Outputs: filtered image, filtered image size, mask size

def top_edge(img_seg, img_size):
    
    # Mask size (depending on img_size) and filtered image size
    mask_size = int(2 * round(0.0025*img_size[1]) + 1)
    filt_size = (img_size[0] - mask_size + 1, img_size[1] - mask_size + 1)
    
    # 3D horizontal mask for quicker convolution
    mask = np.concatenate([np.full(filt_size + (mask_size*(mask_size/2),),  False, dtype=bool),
                           np.full(filt_size + (mask_size*(mask_size/2+1),), True, dtype=bool)], -1)
    
    # 3D image for quicker convolution
    img_conv = np.full(filt_size + (mask_size*mask_size,), False, dtype=bool)
    for kx in range(mask_size*mask_size):
        ix = (int(kx/mask_size), filt_size[0] + int(kx/mask_size))
        jx = (kx % mask_size, filt_size[1] + kx % mask_size)
        img_conv[:,:,kx] = img_seg[ix[0]:ix[1], jx[0]:jx[1]]
    
    # Threshold for convolution
    filt_k = 0.8*mask_size*mask_size
    
    # Binary convolution (Top edge of the segmented color)
    img_filt = np.sum(np.equal(img_conv, mask), 2) > filt_k
    
    return img_filt, filt_size, mask_size



### Function to find the corners of the top edge and consequently the image angle and scale
### Inputs: filtered image
### Outputs: top edge corners, image angle, image scale (size of edge)

def image_properties(img_filt):
    
    # Top corners of the segmented color
    points_seg = ([0,0],[0,0])
    points_seg[0][0] = np.min(np.where(np.any(img_filt,0)))
    points_seg[0][1] = int(round(np.mean(np.where(img_filt[:,points_seg[0][0]]))))
    points_seg[1][0] = np.max(np.where(np.any(img_filt,0)))
    points_seg[1][1] = int(round(np.mean(np.where(img_filt[:,points_seg[1][0]]))))
    
    # Tilte angle and top edge size
    points_ang = -atan2(points_seg[1][1] - points_seg[0][1], points_seg[1][0] - points_seg[0][0])
    points_dist = sqrt((points_seg[1][1] - points_seg[0][1])**2 + (points_seg[1][0] - points_seg[0][0])**2)
    
    return points_seg, points_ang, points_dist



### Function to find the corners related to the text position
### Inputs: top edge corners, image angle, image scale
### Outputs: points of text position

def text_position(points_seg, points_ang, points_dist):
    
    # Constants related to the OK text position (from left-top corner)
    text_k = np.multiply([0.15, 0.10, 0.49, 0.28], points_dist)
    
    # Relative position of the OK text (from left-top corner)
    text_dist = np.full((4,2), 0, dtype='float')
    text_dist[0,:] = np.multiply(text_k[0], [cos(points_ang),      -sin(points_ang)])
    text_dist[1,:] = np.multiply(text_k[1], [cos(points_ang+pi/2), -sin(points_ang+pi/2)])
    text_dist[2,:] = np.multiply(text_k[2], [cos(points_ang),      -sin(points_ang)])
    text_dist[3,:] = np.multiply(text_k[3], [cos(points_ang+pi/2), -sin(points_ang+pi/2)])
    
    # Absolute position of the OK text (using left-top corner)
    text_pos = ([0,0,0,0],[0,0,0,0])
    text_pos[0][0] = int(round(points_seg[0][0] + text_dist[0,0] + text_dist[1,0]))
    text_pos[1][0] = int(round(points_seg[0][1] + text_dist[0,1] + text_dist[1,1]))
    text_pos[0][1] = int(round(points_seg[0][0] + text_dist[0,0] + text_dist[3,0]))
    text_pos[1][1] = int(round(points_seg[0][1] + text_dist[0,1] + text_dist[3,1]))
    text_pos[0][2] = int(round(points_seg[0][0] + text_dist[2,0] + text_dist[3,0]))
    text_pos[1][2] = int(round(points_seg[0][1] + text_dist[2,1] + text_dist[3,1]))
    text_pos[0][3] = int(round(points_seg[0][0] + text_dist[2,0] + text_dist[1,0]))
    text_pos[1][3] = int(round(points_seg[0][1] + text_dist[2,1] + text_dist[1,1]))
    
    return text_pos



### Function to correct the found text (resizes, rotations and crops)
### Inputs: HSV image, text position, mask size, image angle, image scale, default scale
### Outputs: segmented text image, text image size

def correct_text(hsv, text_pos, mask_size, points_ang, points_dist, default_dist):
    
    # Short image of the OK text
    text_ix = (min(text_pos[1]) + mask_size-1, max(text_pos[1]) + mask_size-1)
    text_jx = (min(text_pos[0]) + mask_size-1, max(text_pos[0]) + mask_size-1)
    text_full = hsv[text_ix[0]:text_ix[1], text_jx[0]:text_jx[1], 2]
    
    # Rotation af points_ang
    text_rot = imutils.rotate(text_full, -180/pi*(pi+points_ang))
    
    # Resize to a default scale
    text_size_old = (int(round(default_dist/points_dist*text_rot.shape[0])), 
                     int(round(default_dist/points_dist*text_rot.shape[1])))
    text_resize = cv2.resize(text_rot, (text_size_old[1], text_size_old[0]))
    
    # Crop to a default size (26,52)
    text_size = (30,60)
    text = text_resize[(text_size_old[0]/2-text_size[0]/2):(text_size_old[0]/2+text_size[0]/2), 
                       (text_size_old[1]/2-text_size[1]/2):(text_size_old[1]/2+text_size[1]/2)]
    
    # Segmentation of the OK text
    text_seg = text > 179
    
    return text_seg, text_size



### Function to generate the template of an OK text (work only with 720p input image)
### Inputs: []
### Outputs: OK image template, OK size

def create_mask():
    
    # Mask for OK
    ok = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    
    # Size of OK mask and final result
    ok_size = ok.shape
    
    return ok, ok_size



### Function to perform a binary convolution between the segmented text and the mask
### Inputs: segmented text, text size, mask, mask size
### Outputs: convolution result image

def ok_convolution(text_seg, text_size, ok, ok_size):
    
    #
    res_size = (text_size[0] - ok_size[0] + 1, text_size[1] - ok_size[1] + 1)
    
    # Threshold for convolution (limited between 200~500)
    res_k = max([200, min([500, np.sum(text_seg), np.sum(ok)])])
    
    # Simple binary convolution
    res_filt = np.full(res_size, 0, dtype=float)
    for ix in range(res_size[0]):
        for jx in range(res_size[1]):
            res_filt[ix, jx] = float(np.sum( (text_seg[ix:(ix+ok_size[0]), jx:(jx+ok_size[1])]) & ok )) / res_k
    
    return res_filt



### Function to verify the presence of OK in the input file
### Inputs: input file
### Outputs: status of OK

def ok_test(input_file):
    
    global hsv, img_seg, img_filt, points_seg, points_ang, points_dist, text_pos, text_seg, ok, res_filt, output_status
    
    # Read the image in HSV
    hsv = read_hsv(input_file)
    
    # Segment the green color of the HSV image
    img_seg, img_size = green_segmentation(hsv)
    
    # Find the top edge of the segmentation
    img_filt, filt_size, mask_size = top_edge(img_seg, img_size)
    
    # Find the corners of the top edge and the full image angle and scale
    points_seg, points_ang, points_dist = image_properties(img_filt)
    
    # Set a default scale for image (work well for 720p resolution)
    default_dist = 180
    
    # Find the points that delimit the text position
    text_pos = text_position(points_seg, points_ang, points_dist)
    
    # Find the corrected image of the text (cropped, rotated and resized)
    text_seg, text_size = correct_text(hsv, text_pos, mask_size, points_ang, points_dist, default_dist)
    
    # Generate a mask for OK text (work well for 720p resolution)
    ok, ok_size = create_mask()
    
    # Perform a convolution with the segmented text and the mask
    res_filt = ok_convolution(text_seg, text_size, ok, ok_size)
    
    # Final result (if any convolution result is greater then 0.7)
    output_status = np.max(res_filt) > 0.7
    
    return (output_status,)



# Main function

if __name__ == "__main__":
    
    # Path of input file
    input_file = '/home/rafael/Github/icts_pdi_challenge/1.jpg'
    
    # Verify the OK presence, returning (True,) or (False,)
    output_status = ok_test(input_file)

import os
import numpy as np
import sys
import time # time the execution time

import caffe
import cv2

model = '/home/yangcheng/workspace/VPGNet/caffe/models/vpgnet-novp/deploy.prototxt'
pretrained = '/home/yangcheng/workspace/VPGNet/caffe/models/vpgnet-novp/snapshots/split_iter_80000.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(model, pretrained, caffe.TEST)

img = caffe.io.load_image('1.jpg')

t = time.time()

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))
transformed_img = transformer.preprocess('data', img) # swap R, B channel, the final input to the network should be RGB
print transformed_img.shape
net.blobs['data'].data[...] = transformed_img

#print(transformed_img)
t1 = time.time()
for i in range(1):
    net.forward()
    # for j in range(1000000): # mimic post process
    #     pass
print "forward propagation time: ", time.time() - t1

dt = time.time() - t
print "Timing ends! Process time:", dt


img = cv2.imread('example.png')

obj_mask = net.blobs['binary-mask'].data
print net.blobs
print obj_mask.shape
print transformed_img.shape

x_offset_mask = 4 # offset to align output with original pic: due to padding
y_offset_mask = 4

masked_img = img.copy()
mask_grid_size = img.shape[0] / obj_mask.shape[2]
tot = 0
for i in range(120):
    for j in range(160):
        mapped_value =  int(obj_mask[0, 0, i, j] * 255)
        obj_mask[0, 0, i, j] = mapped_value

        mapped_value =  int(obj_mask[0, 1, i, j] * 255)
        obj_mask[0, 1, i, j] = mapped_value

        #if mapped_value > 100:
        #    masked_img[(i+y_offset_mask)*mask_grid_size : (i+1+y_offset_mask)*mask_grid_size + 1, (j+x_offset_mask)*mask_grid_size : (j+x_offset_mask+1)*mask_grid_size + 1]\
        #     = (mapped_value, mapped_value, mapped_value) # mask with white block

#small_mask = obj_mask[0, 1, ...]
#resized_mask = cv2.resize(small_mask, (640, 480))
#translationM = np.float32([[1, 0, x_offset_mask*mask_grid_size], [0, 1, y_offset_mask*mask_grid_size]])
#print translationM
#resized_mask = cv2.warpAffine(resized_mask, translationM, (640, 480)) # translate (shift) the image
#cv2.imwrite(workspace_root + 'mask.png', resized_mask)
#cv2.imwrite(workspace_root + 'masked.png', masked_img)

classification = net.blobs['multi-label'].data
print classification.shape
classes = []


# create color for visualizing classification
def color_options(x):
    return {
        1: (0, 255, 0), # green color
        2: (255, 0, 0), # blue
        3: (0, 0, 255), # red
        4: (0, 0, 0)
    }[x]

# visualize classification
y_offset_class = 1 # offset for classification error
x_offset_class = 1
grid_size = img.shape[0]/60
for i in range(60):
    classes.append([])
    for j in range(80):
        max_value = 0
        maxi = 0
        for k in range(64):
            if classification[0, k, i, j] > max_value:
                max_value = classification[0, k, i, j]
                maxi = k

        classes[i].append(maxi)
        if maxi != 0:
	    #print maxi
            pt1 = ((j + y_offset_class)*grid_size, (i+x_offset_class)*grid_size)
            pt2 = ((j + y_offset_class)*grid_size+grid_size, (i+x_offset_class)*grid_size+grid_size)
            # print maxi
            cv2.rectangle(img, pt1, pt2, color_options(maxi), 2)
            if maxi not in [1, 2, 3, 4]:
                print "ERROR OCCURRED: an unknown class detected!"

cv2.imwrite("result_py.png", img) # ISSUE1: the image BGR channel VS RGB


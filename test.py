import numpy as np
import keras
import cv2
from dataset import DataGenerator
from network import Network
import os
import glob
from pathlib import Path
import math
import matplotlib.pyplot as plt


input_shape=(512,384,3)
nb_classes=18
path_ref="/home/aimenext/cuongdx/tripletloss/all_classified"
path_test="/home/aimenext/cuongdx/tripletloss/data_train"

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)

def compute_dist(a, b):
	return np.sum(np.square(a - b))

def get_test_imgs(path):
    path_imgs=[]
    for root,dir,files in os.walk(path):
        if len(files)>0:
            file=np.random.choice(files)
            file_path=os.path.join(root,file)
            path_imgs.append(file_path)
    return path_imgs

def pre_process(img_path):
    img = cv2.imread(img_path)
    img = DataGenerator.pre_process(img, size=(input_shape[1], input_shape[0]))
    return img

def get_ref_img(path,cls):
    folder=os.path.join(path,str(cls))
    files=glob.glob(os.path.join(folder,"*"))
    # print(files)
    file=np.random.choice(files)
    return file

def get_ref_imgs(path_ref,test_imgs):
    ref_path=[]
    for path in test_imgs:
        # print("path: ",path)
        parent=str(Path(path).parents[0])
        cls=parent.split(os.sep)[-1]
        # print("path ref: ",path_ref,cls)
        file=get_ref_img(path_ref,cls)
        ref_path.append(file)
    return ref_path

def test_images(network,test_path,ref_path):
    test=get_test_imgs(test_path)
    ref=get_ref_imgs(ref_path,test)
    test_imgs=[]
    ref_imgs=[]
    # print(test[0])
    # print(ref[0])
    # print(test[0])
    # print(ref[0])
    for file in test:
        img=pre_process(file)
        test_imgs.append(img)
    for file in ref:
        img=pre_process(file)
        ref_imgs.append(img)
    ref_imgs=np.array(ref_imgs)
    test_imgs=np.array(test_imgs)
    # print(ref_imgs)
    ref_pred=network.predict_image(ref_imgs)
    test_pred=network.predict_image(test_imgs)
    for i in range(nb_classes):
        distance=[]
        # fig = plt.figure(figsize=(16, 2))
        # subplot = fig.add_subplot(1, nb_classes + 1, 1)
        # plt.axis("off")
        # plotidx = 2
        # image=cv2.imread(test[i])
        # plt.imshow(image, vmin=0, vmax=1, cmap='Greys')
        # subplot.title.set_text("Test image")

        for j in range(nb_classes):
            dist=compute_dist(ref_pred[i],test_pred[j])
            distance.append(dist)

        #     # Draw
        #     subplot = fig.add_subplot(1, nb_classes + 1, plotidx)
        #     plt.axis("off")
        #     img_ref=cv2.imread(ref[j])
        #     plt.imshow(img_ref, vmin=0, vmax=1, cmap='Greys')
        #     subplot.title.set_text(("Class {0}\n{1:.3e}".format(ref, dist)))
        #     plotidx += 1
        # plt.show()
        arg_min=np.argmin(distance)
        print("actual class: ",get_class(test[i]))
        print("predicted class: ",get_class(ref[arg_min]))
        print(distance)
        # print(str(Path(test[i]).parents[0]).split(os.sep)[-1])
        print("--"*40)

def get_class(folder):
    # print(folder)
    path=str(Path(folder).parents[0])
    cls=path.split(os.sep)[-1]
    return int(cls)

# nb_classes = 10
# img_rows, img_cols = 28, 28
# input_shape = (img_rows, img_cols, 1)
#
# network=build_network(input_shape,nb_classes)
# network_train = build_model(input_shape, network)
# network_train.load_weights("step-15000_loss-1.7682428359985352.h5")
#
# for i in range(10):
# 	featured_img = network.predict(x_test_origin[i].reshape(1,28,28,1))
# 	print(y_test_origin[i],np.argmax(softmax(featured_img)))


# network = Network(nb_classes, input_shape)
# network.network_train.load_weights("/home/aimenext/cuongdx/tripletloss/models/epoch-88_loss-0.00822.h5")
# # for layer in network.base_network.layers:
# #     weights=layer.get_weights()
# #     for weight in weights:
# #         print(weight)
#
#
# test_images(network,path_test,path_ref)

def find_max_group(gt_group, pred_group):
    freq = dict((k, [0]*len(pred_group)) for k in gt_group)
    for idx, (key, values) in enumerate(pred_group.items()):
        for v in values:
            for id_gt, (key_gt, values_gt) in enumerate(gt_group.items()):
                if v in values_gt:
                    freq[key_gt][idx] += 1
    rs=dict((k, max(v)) for k, v in freq.items())
    print(freq)
    print(rs)

a={'a':[1,2,3,4],'b':[5,6,7,8]}
x={'s':[1,2,3,6],'d':[4,5],'c':[7,8]}
find_max_group(a,x)
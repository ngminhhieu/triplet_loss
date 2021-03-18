import cv2
import numpy as np
from keras.models import Sequential
from keras.utils import Sequence
import os
from network import Network
from keras.optimizers import Adam


class DataGenerator(Sequence):
    def __init__(self, x_set, batch_size, width, height, channel, nb_hardest, nb_normal, net, nb_classes, classes,
                 focus_folder=None,
                 ratio_sample=4, ratio_probs=5):
        self.x_set = x_set
        self.batch_size = batch_size
        self.net = net
        self.nb_classes = nb_classes
        self.classes = classes
        self.ratio = ratio_sample
        self.width = width
        self.height = height
        self.channel = channel
        self.nb_hardest = nb_hardest
        self.nb_normal = nb_normal
        # self.on_epoch_end()
        self.focus_folder = focus_folder
        self.keys = list(x_set.keys())
        self.ratio_probs = ratio_probs
        self.probs = self.get_probs()

    def get_probs(self):
        keys = self.keys
        focus_key = self.focus_folder
        prob = 1 / (len(keys) + (self.ratio_probs - 1) * len(focus_key))
        probs = []
        for key in keys:
            if key in focus_key:
                probs.append(prob * self.ratio_probs)
            else:
                probs.append(prob)
        return probs

    def __len__(self):
        nb_elements = sum(len(self.x_set[k]) for k in self.x_set)
        self.on_epoch_end()
        return int(np.floor(nb_elements / self.batch_size))

    def __getitem__(self, idx):
        return self.get_batch_hard()

    def get_batch_random(self):
        triplets = [np.zeros((self.batch_size * self.ratio, self.height, self.width, self.channel)) for i in range(3)]
        for i in range(self.batch_size * self.ratio):
            classes_copy = self.classes.copy()
            anchor_class = np.random.choice(self.keys, p=self.probs)
            nb_sample_ap = len(self.x_set[anchor_class])
            [idx_A, idx_P] = np.random.choice(nb_sample_ap, size=2, replace=False)
            classes_copy.remove(anchor_class)
            negative_class = np.random.choice(classes_copy)
            # print("class: ", negative_class, anchor_class,len(classes_copy))
            nb_sample_negative = len(self.x_set[negative_class])
            idx_N = np.random.randint(0, nb_sample_negative)

            data_A, data_P, data_N = self.get_data(anchor_class, negative_class, idx_A, idx_P, idx_N)
            triplets[0][i, :, :, :] = data_A
            triplets[1][i, :, :, :] = data_P
            triplets[2][i, :, :, :] = data_N
        return triplets

    def get_batch_hard(self):
        random_batch = self.get_batch_random()
        loss_A = self.net.predict(random_batch[0])
        loss_P = self.net.predict(random_batch[1])
        loss_N = self.net.predict(random_batch[2])
        # Compute d(A,P)-d(A,N)
        batch_loss = np.sum(np.square(loss_A - loss_P), axis=1) - np.sum(np.square(loss_A - loss_N), axis=1)
        hardest_loss = np.argsort(batch_loss)[::-1][:self.nb_hardest]
        # print("hardest: ", hardest_loss)
        normal_loss = np.random.choice(np.delete(np.arange(self.batch_size * self.ratio), hardest_loss), self.nb_normal,
                                       replace=False)
        selection = np.append(hardest_loss, normal_loss)
        triplets = [random_batch[0][selection, :, :, :], random_batch[1][selection, :, :, :],
                    random_batch[2][selection, :, :, :]]
        return triplets

    @staticmethod
    def resize_img(img, size):
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        # should be RGB order
        img = in_img.copy().astype(np.float32)
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

    @staticmethod
    def pre_process(img, size):
        img = DataGenerator.resize_img(img, size)
        img = DataGenerator.normalizeMeanVariance(in_img=img)
        return img

    def get_data(self, anchor_class, negative_class, idx_A, idx_P, idx_N):
        # load
        img_A = cv2.imread(self.x_set[anchor_class][idx_A])
        img_P = cv2.imread(self.x_set[anchor_class][idx_P])
        img_N = cv2.imread(self.x_set[negative_class][idx_N])
        # print(self.x_set[anchor_class][idx_A],self.x_set[anchor_class][idx_P],self.x_set[negative_class][idx_N])

        # resize
        size = (self.width, self.height)
        img_A = DataGenerator.pre_process(img_A, size)
        img_P = DataGenerator.pre_process(img_P, size)
        img_N = DataGenerator.pre_process(img_N, size)
        return img_A, img_P, img_N

    def on_epoch_end(self):
        print("epoch end, ratio prob: ", self.ratio_probs)
        for cls in self.x_set:
            np.random.shuffle(self.x_set[cls])
        keys = self.keys
        np.random.shuffle(keys)
        self.x_set = dict((k, self.x_set[k]) for k in keys)
        self.keys=list(self.x_set.keys())
        self.probs = self.get_probs()

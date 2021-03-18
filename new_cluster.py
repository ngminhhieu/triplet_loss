import numpy as np
import keras
import cv2
from network import Network, OPTION
import os
import glob
from pathlib import Path
import math
from dataset import DataGenerator
from yellowbrick.cluster import KElbowVisualizer
import shutil
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from datetime import datetime
import json


def load_model(weight_path, nb_classes, input_shape):
    network = Network(nb_classes, input_shape)
    network.network_train.load_weights(weight_path)
    return network


def load_imgs(path, size):
    files = glob.glob(os.path.join(path, "*"))
    imgs = []
    for file in files:
        img = cv2.imread(file)
        img = DataGenerator.pre_process(img, size)
        imgs.append(img)
    return files, imgs


def get_test_imgs(path, size):
    path_imgs = []
    for root, dir, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            path_imgs.append(file_path)
    imgs = []
    for file in path_imgs:
        img = cv2.imread(file)
        img = DataGenerator.pre_process(img, size)
        imgs.append(img)
    return path_imgs, imgs


def get_group(path):
    dict_ = {}
    for p in path:
        bn = os.path.basename(p)
        cls = get_class(p)
        if cls not in dict_:
            dict_[cls] = [bn]
        else:
            dict_[cls].append(bn)
    return dict_


def predict_imgs(network, imgs, batch_size=8, option=None):
    # preds = network.predict_image(imgs)
    nb_steps = int(np.ceil(imgs.shape[0] / 8))
    result = []
    for i in range(nb_steps):
        preds = network.predict_imgs(imgs[batch_size * i:min(batch_size * (i + 1), imgs.shape[0])], option=option)
        for pred in preds:
            result.append(pred)
    result = np.array(result)
    result = result.reshape(imgs.shape[0], -1)
    return np.array(result)


def get_nb_clusters(X, range_clusters):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=range_clusters)
    visualizer.fit(X)
    print("number of clusters: ", visualizer.elbow_value_)
    return visualizer.elbow_value_


def cluster_kmean(X, nb_cluster, files):
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X)
    targetdir = "triplet_kmean_" + os.sep + str(nb_cluster)
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("labels: ", set(kmeans.labels_))
    for i in set(kmeans.labels_):
        path = os.path.join(targetdir, str(i))
        if not os.path.exists(path):
            os.makedirs(path)
    for i, m in enumerate(kmeans.labels_):
        print("Copy: %s / %s" % (i, len(kmeans.labels_)))
        fn = os.path.basename(files[i])
        shutil.copy(files[i], os.path.join(os.path.join(targetdir, str(m)), fn))


def main_kmean(path_imgs, weight_path, nb_class=18, input_shape=(512, 384, 3)):
    files, imgs = load_imgs(path_imgs, size=(input_shape[1], input_shape[0]))
    network = load_model(weight_path, nb_class, input_shape)
    np_imgs = np.array(imgs)
    preds = predict_imgs(network, np_imgs)
    nb_clusters = get_nb_clusters(preds, range_clusters=(5, 25))
    cluster_kmean(preds, nb_clusters, files)


def draw_matrix_similarity(preds, matrix=None):
    # draw matrix similarity
    if matrix is None:
        matrix = 1 - pairwise_distances(preds, metric="cosine")
    fig, ax = plt.subplots(figsize=(matrix.shape))
    # cax = ax.matshow(matrix, interpolation='nearest')
    # ax.grid(True)
    list = []
    # plt.title(' Score similarity matrix')
    # plt.xticks(range(10), list)
    # plt.yticks(range(10), list)
    # fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
    # plt.show()

    # # cmap = cm.get_cmap('Greens')
    # # cmap = cm.get_cmap('YlGnBu')
    # cmap = cm.get_cmap('RdYlGn')
    # cax = ax.matshow(matrix, interpolation='nearest', cmap=cmap)
    #


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_class(folder):
    # print(folder)
    path = str(Path(folder).parents[0])
    cls = path.split(os.sep)[-1]
    return cls


def list_contain_list(sublist, list):
    return all(sub in list for sub in sublist)


def merger_group(dict_group):
    del_key = []
    for key, values in dict_group.items():
        for k, v in dict_group.items():
            if key != k:
                if len(values) < len(v) and list_contain_list(values, v):
                    del_key.append(key)
                elif len(v) < len(values) and list_contain_list(v, values):
                    del_key.append(k)
    for key in set(del_key):
        del dict_group[key]
    return dict_group


def folder_to_int(list):
    list.sort()
    dict_ = {}
    for idx, l in enumerate(list):
        dict_[l] = idx
    return dict_


def get_similarity_score(feature_1, feature_2):
    return 1 - pairwise_distances(feature_1, feature_2, metric="cosine")


def find_group_from_idx(group_cls, idx):
    for k, v in group_cls.items():
        if idx in v:
            return k
    return -1


def find_group(test_feature, ref_feature, group_cls, thresh, classify=False, average=False):
    result = {}
    for idx, feature in enumerate(ref_feature):
        group = find_group_from_idx(group_cls, idx)
        if classify:
            score = compute_dist(test_feature, feature)
        else:
            score = get_similarity_score([test_feature], [feature])
        add_to_dict(result, group, score)
    for k, v in result.items():
        if average:
            v = np.sum(v) / len(v)
            result[k] = v
        else:
            if classify:
                m = np.min(v)
            else:
                m = np.max(v)
            result[k] = m
    sorted_key = sorted(result.keys(), key=lambda k: result[k], reverse=not classify)
    group = -1
    if classify:
        if result[sorted_key[0]] <= thresh:
            group = sorted_key[0]
    else:
        if result[sorted_key[0]] >= thresh:
            group = sorted_key[0]

    return update_group(test_feature, ref_feature, group_cls, group)


def create_new_group(nb_group):
    return "group_%s" % (nb_group + 1)


def update_group(test_feature, ref_feature, group_cls, id_group, append_current_group=False):
    if id_group == -1:
        id_group = create_new_group(len(group_cls))
        group_cls[id_group] = [len(ref_feature)]
        ref_feature = np.vstack((ref_feature, test_feature))
    elif id_group != -1 and append_current_group:
        group_cls[id_group].append(len(ref_feature))
        ref_feature = np.vstack((ref_feature, test_feature))

    return id_group, ref_feature


def add_to_dict(dict_, key, v=1, append=1):
    if key not in dict_:
        if append == 1:
            dict_[key] = [v]
        else:
            dict_[key] = v
    else:
        if append == 1:
            dict_[key].append(v)
        elif append == 0:
            dict_[key] += v
        else:
            dict_[key].extend(v)


def compute_dist(a, b):
    return np.sum(np.square(a - b))


# def result_to_json(result, accuracy):
#     result = {
#         'Fail': [
#             {
#                 'file name': fn,
#                 'ground truth': gt,
#                 'prediction': pred
#             }
#             for fn, gt, pred in result],
#         "Accuracy:": {
#             "correct file": accuracy[0],
#             "total file": accuracy[1],
#             "accuracy": np.round(accuracy[0] / accuracy[1], 2) if accuracy[1] != 0 else 0
#         }
#     }
#     return result

def result_to_json(result, accuracy, list_files):
    dict_ = {}
    rs = []
    for key, dict_values in result.items():
        gt = ': ' + key
        item = []
        for key_pred, values in dict_values.items():
            incorrect = {}
            incorrect['prediction'] = key_pred
            list_values = []
            for value in values:
                list_values.append(list_files[value])
            incorrect['files name'] = list_values
            item.append(incorrect)
        rs.append({'ground truth': key, 'incorrect': item})

    correct = 0
    total = 0
    for k, v in accuracy.items():
        correct += v[0]
        total += v[1]

    dict_["Fail"] = rs
    dict_["Accuracy:"] = {
        "correct file": correct,
        "total file": total,
        "accuracy": np.round(correct / total, 2) if total != 0 else 0
    }
    dict_["Detail"] = accuracy
    return dict_


def test_cluster(preds, ref_features, group_cls, thresh=0.9, thresh_cluster=0.05):
    # if eval:
    #     group_cls = get_group(files)
    result_dict = dict((k, []) for k in group_cls)
    for idx, pred in enumerate(preds):
        group, ref_features = find_group(pred, ref_features, group_cls, thresh)
        max_diff, min_diag, mat = cal_threshold(ref_features, group_cls)
        thresh = min_diag - thresh_cluster
        print(thresh)
        add_to_dict(result_dict, group, idx)
    # print(result_dict)
    return result_dict


def test_classify(preds, ref_features, group_cls, thresh=1):
    result_dict = dict((k, []) for k in group_cls)
    for idx, pred in enumerate(preds):
        group, ref_features = find_group(pred, ref_features, group_cls, thresh, classify=True)
        add_to_dict(result_dict, group, idx)
    print(result_dict)
    return result_dict


def find_group_with_old_model(ref_group, gt_group, pred_group):
    freq = dict((k, {}) for k in gt_group)
    # all_key = set(list(gt_group.keys()) + list(ref_group.keys()))
    # incorrect {'group':[[group_fail':number],[...]]
    incorrect = dict((k, {}) for k in gt_group)
    # detail_incorrect  {'group':{group_fail':[idx_file1, idx_file2,...]],{...}}
    detail_incorrect = dict((k, {}) for k in gt_group)
    for idx, (key, values) in enumerate(pred_group.items()):
        # if key in ref_group.keys():
        #     continue
        for v in values:
            for id_gt, (key_gt, values_gt) in enumerate(gt_group.items()):
                if v in values_gt:
                    add_to_dict(freq[key_gt], key, v)
                    # freq[key_gt][idx] += 1
    rs = {}
    for key, values_dict in freq.items():
        number_files = [len(v) for k, v in values_dict.items() if k not in ref_group]
        rs[key] = max(number_files) if len(number_files) > 0 else 0
        first = 0
        for idx, (k, values) in enumerate(values_dict.items()):
            if len(values) == rs[key] and k not in freq and first == 0:
                first = 1
                continue
            add_to_dict(incorrect[key], k, len(values), append=0)
            add_to_dict(detail_incorrect[key], k, values, 2)

    # keys_pred = list(pred_group.keys())
    # for key in ref_group:
    #     values=pred_group.get(key,{})
    #     for value in values:
    #         id_group=find_group_from_idx(gt_group,value)
    #         add_to_dict(incorrect[id_group], key,append=False)
    #         add_to_dict(detail_incorrect[id_group],key,value,append=True)

    # for i, values in freq.items():
    #     idx_max = np.argmax(values)
    #     for idx, value in enumerate(values):
    #         if idx_max == idx:
    #             continue
    #         if value > 0:
    #             add_to_dict(incorrect[i],keys_pred[idx],value,append=False)
    #             add_to_dict(detail_incorrect[i],keys_pred[idx],value,append=True)
    #             # incorrect[i].append([{keys_pred[idx]:value}])
    for k, v in rs.items():
        rs[k] = [v, len(gt_group[k])]
    incorrect = dict((k, v) for k, v in incorrect.items() if len(v) > 0)
    detail_incorrect = dict((k, v) for k, v in detail_incorrect.items() if len(v) > 0)
    return rs, incorrect, detail_incorrect


def find_group_with_current_model(ref_group, gt_group, pred_group):
    result = dict((k, 0) for k in gt_group)
    all_key = set(list(gt_group.keys()) + list(ref_group.keys()))
    # incorrect {'group':{group_fail':number,{...}}
    incorrect = dict((k, {}) for k in all_key)
    # detail_incorrect  {'group':{group_fail':[idx_file1, idx_file2,...]],{...}}
    detail_incorrect = dict((k, {}) for k in all_key)
    exist_gr = dict((k, v) for k, v in gt_group.items() if k in ref_group)
    not_exist_gr = dict((k, v) for k, v in gt_group.items() if k not in ref_group)

    for key, values in exist_gr.items():
        for v in values:
            gr_pred = find_group_from_idx(pred_group, v)
            if gr_pred == key:
                add_to_dict(result, key, append=0)
            else:
                add_to_dict(incorrect[key], gr_pred, append=0)
                add_to_dict(detail_incorrect[key], gr_pred, v)
    result_not_exist, incorrect_not_exist, detail_incorrect_not_exist = find_group_with_old_model(ref_group, not_exist_gr, pred_group)
    incorrect = dict((k, v) for k, v in incorrect.items() if len(v) > 0)
    detail_incorrect = dict((k, v) for k, v in detail_incorrect.items() if len(v) > 0)
    for k, v in result.items():
        result[k] = [v, len(gt_group[k])]
    result.update(result_not_exist)
    incorrect.update(incorrect_not_exist)
    detail_incorrect.update(detail_incorrect_not_exist)
    return result, incorrect, detail_incorrect

def split_path(list_imgs, size):
    path_imgs = []
    imgs = []
    for file in list_imgs:
        img = cv2.imread(file['path'])
        img = DataGenerator.pre_process(img, size)
        imgs.append(img)
        path_imgs.append(file['path'])
    return path_imgs, imgs

def test(list_imgs, path_ref, weight_path, nb_class=18, input_shape=(512, 384, 3), save_path="", eval=False, option=None, classify=False, thresh_cluster=0.05,
         thresh_classify=0.2):
    files_test, imgs_test = split_path(list_imgs, size=(input_shape[1], input_shape[0]))
    files_ref, imgs_ref = get_test_imgs(path_ref, size=(input_shape[1], input_shape[0]))
    network = load_model('epoch-90_loss-0.00748.h5', nb_class, input_shape)

    np_imgs_ref = np.array(imgs_ref)
    # if classify:
    #     option=OPTION[1]
    preds_ref = predict_imgs(network, np_imgs_ref, option=option)
    group_cls = group_from_files(files_ref)
    np_imgs_test = np.array(imgs_test)
    preds_test = predict_imgs(network, np_imgs_test, option=option)

    max_diff, min_diag, mat = cal_threshold(preds_ref, group_cls)
    abs_test_path = os.path.abspath(path_imgs_test)
    folder_test = abs_test_path.split(os.sep)[-1]
    time = datetime.now()
    format_time = time.strftime("%m_%d_%Y")
    model_name = os.path.basename(weight_path)
    save_path = "%s/%s_%s_%s" % (save_path, model_name, format_time, option)
    if classify:
        result = test_classify(preds_test, preds_ref, group_cls, thresh=thresh_classify)
        save_path = "%s_%s_classify-%s_eval_%s_thresh_%s" % (save_path, folder_test, classify, eval, thresh_classify)
    else:
        result = test_cluster(preds_test, preds_ref, group_cls, min_diag - thresh_cluster)
        save_path = "%s_%s_init-thresh_%s_thresh_%s" % (save_path, folder_test, np.round(min_diag, 3), thresh_cluster)
    create_folder(save_path)

    for key, values in result.items():
        folder = os.path.join(save_path, str(key))
        create_folder(folder)
        for value in values:
            parent = str(Path(files_test[value]).parents[0])
            path = parent.split(os.sep)[-1]
            fn = "%s_%s" % (path, os.path.basename(files_test[value]))
            shutil.copy(files_test[value], os.path.join(folder, fn))

    total = 0
    correct = 0

    rs = []
    acc = {}
    if eval:
        ref_group = group_from_files(files_ref)
        group_cls_gt = group_from_files(files_test)

        acc, incorrect, detail = find_group_with_current_model(ref_group, group_cls_gt, result)
        print(incorrect)
        print(detail)
        json_data = result_to_json(detail, acc, files_test)
        with open(os.path.join(save_path, "result.json"), "w+") as json_file:
            json.dump(json_data, json_file)

    print(acc)
    correct = 0
    total = 0
    for k, v in acc.items():
        correct += v[0]
        total += v[1]
    print("correct {}, total {}".format(correct, total))
    print(save_path)


def main_cosin(path_imgs, weight_path, nb_class=18, input_shape=(512, 384, 3), thresh=0.9, eval=False):
    files, imgs = get_test_imgs(path_imgs, size=(input_shape[1], input_shape[0]))
    if eval:
        group_cls = get_group(files)
    network = load_model(weight_path, nb_class, input_shape)
    np_imgs = np.array(imgs)
    preds = predict_imgs(network, np_imgs)
    draw_matrix_similarity(preds)
    matrix = 1 - pairwise_distances(preds, metric="cosine")
    rows, cols = matrix.shape
    result_dict = {}
    for r in range(rows):
        similar = [r]
        for c in range(cols):
            if matrix[r, c] > thresh and r != c:
                similar.append(c)
        value_in_dict(similar, result_dict)
    print("dict: ", result_dict)
    save_folder = "../output"
    create_folder(save_folder)
    result_dict = merger_group(result_dict)
    for key, values in result_dict.items():
        path = create_folder(os.path.join(save_folder, str(key)))
        for v in values:
            fn = os.path.basename(files[v])
            shutil.copy(files[v], os.path.join(path, fn))

    cnt = 0
    if eval:
        for key, values in result_dict.items():
            gr = []
            for v in values:
                fn = os.path.basename(files[v])
                gr.append(fn)
            gr.sort()
            found = False
            for k, value in group_cls.items():
                value.sort()
                if value == gr:
                    cnt += 1
                    found = True
                    break
            if not found:
                print(gr)
    print(cnt)


def cal_threshold(ref_features, group_cls):
    # ref_features:
    # group_cls:dictionary: folder contain index of img {'abc':[0,1,2],'def':[3,4,5]}

    matrix = 1 - pairwise_distances(ref_features, metric="cosine")

    keys = list(group_cls.keys())
    group_int = folder_to_int(keys)
    mat = np.zeros((len(keys), len(keys)))

    for i in range(len(keys)):
        cls_i = keys[i]
        for j in range(i, len(keys)):
            cls_j = keys[j]
            index = get_combination(group_cls[cls_i], group_cls[cls_j])
            for idx in index:
                mat[group_int[cls_i], group_int[cls_j]] += matrix[idx[0], idx[1]]
            mat[group_int[cls_i], group_int[cls_j]] /= len(index)
            mat[group_int[cls_j], group_int[cls_i]] = mat[group_int[cls_i], group_int[cls_j]]
    # draw_matrix_similarity(preds,mat)
    max_diff = -1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i != j:
                if mat[i, j] > max_diff:
                    max_diff = mat[i, j]
    min_diag = np.min(np.diagonal(mat))
    print("max diff: ", max_diff)
    print("min similarity: ", min_diag)
    return max_diff, min_diag, mat


def cosin_with_groundtruth(path_imgs, weight_path, nb_class=18, input_shape=(512, 384, 3)):
    files, imgs = get_test_imgs(path_imgs, size=(input_shape[1], input_shape[0]))
    network = load_model(weight_path, nb_class, input_shape)
    np_imgs = np.array(imgs)
    preds = predict_imgs(network, np_imgs)
    group_cls = group_from_files(files)
    max_diff, min_diag, mat = cal_threshold(preds, group_cls)
    mat = np.round(mat, 2)
    for i in range(mat.shape[0]):
        print(mat[i])


def get_combination(list_a, list_b):
    rs = []
    if list_a == list_b:
        if len(list_a) == 1:
            return [[list_a[0], list_a[0]]]
        for i in range(len(list_a)):
            for j in range(i + 1, len(list_a)):
                rs.append([list_a[i], list_a[j]])
    else:
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                rs.append([list_a[i], list_b[j]])
    return rs


def group_from_files(files):
    group_cls = {}
    for idx, file in enumerate(files):
        cls = get_class(file)
        if cls not in group_cls:
            group_cls[cls] = [idx]
        else:
            group_cls[cls].append(idx)
    return group_cls


def value_in_dict(values, dict_):
    found = False
    # print(values,dict_)
    for v in dict_.values():
        if all(value in v for value in values):
            found = True
            break
    if not found:
        dict_[len(dict_)] = values


# def test_main():
#     new = 0
#     if new:
#         weight_path = "../models/03_16_2020_0.0005/epoch-150_loss-0.00621.h5"
#         path_imgs_ref = "../data/ref_data_new"
#     else:
#         weight_path = "../models/0309/epoch-86_loss-0.00906.h5"
#         path_imgs_ref = "../data/ref_data_old"
#     nb_classes = 18
#     input_shape = (512, 384, 3)
#     # thresh_cluster = 0.05
#     # thresh_classify = 0.2

#     # test
#     path_imgs_test = "../data/data_test/new_group_0318"
#     path_imgs_test = "/home/aimenext/cuongdx/tripletloss/data/data_test/new_cate_2"
#     # weight_path = ""
#     # path_imgs_ref = ""
#     # thresh_cluster = 0
#     # thresh_classify = 0
#     # main_cosin(path_imgs, weight_path,thresh=0.75,eval=False)

#     # cal threshold
#     # cosin_with_groundtruth(path_imgs_ref, weight_path)
#     test(path_imgs_test, path_imgs_ref, weight_path, nb_classes, input_shape, "../result", eval=True, option=OPTION[1], classify=False)


# def main(path_imgs_test="../data/data_test/new_group_0318",
#          weight_path="",
#          path_imgs_ref="",
#          save_dir="../result",
#          thresh_cluster=0,
#          thresh_classify=0,
#          classify=False,
#          input_shape=(512, 384, 3),
#          nb_classes=18
#          ):
#     test(path_imgs_test, path_imgs_ref, weight_path, nb_classes, input_shape, save_dir, eval=True, option=OPTION[1], classify=False, thresh_cluster=thresh_cluster,
#          thresh_classify=thresh_classify)
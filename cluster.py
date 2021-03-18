import numpy as np
import keras
import cv2
from network import Network, OPTION
import os
import glob
from pathlib import Path
import math
from dataset import DataGenerator
# from yellowbrick.cluster import KElbowVisualizer
import shutil
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from datetime import datetime
import json
import csv
from collections import OrderedDict
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score


def load_model(weight_path, nb_classes, input_shape):
    network = Network(nb_classes, input_shape)
    # network.network_train.load_weights(weight_path)
    return network


def load_imgs(path, size):
    files = glob.glob(os.path.join(path, "*"))
    imgs = []
    for file in files:
        img = cv2.imread(file)
        img = DataGenerator.pre_process(img, size)
        imgs.append(img)
    return files, imgs


def get_test_imgs(path_list, size):
    path_imgs = []
    for path in path_list:
        for root, dir, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                path_imgs.append(file_path)
    imgs = []
    for file in path_imgs:
        # print(file)
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

def result_to_json(result, accuracy, list_files,threshold=0.5):
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
    dict_["Threshold"]=threshold
    dict_["Fail"] = rs
    dict_["Accuracy:"] = {
        "correct file": correct,
        "total file": total,
        "accuracy": np.round(correct / total, 2) if total != 0 else 0
    }
    dict_["Detail"] = accuracy
    return dict_


def test_cluster(preds, ref_features, group_cls, thresh, thresh_cluster=0.05):
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


def test(path_imgs_test, path_ref, network, nb_class=18, input_shape=(960, 720, 3), save_path="", eval=False, option=None, classify=False,weight_path=None,thresh_classify=1,thresh_cluster=0.05):
    files_test, imgs_test = get_test_imgs(path_imgs_test, size=(input_shape[1], input_shape[0]))
    files_ref, imgs_ref = get_test_imgs(path_ref, size=(input_shape[1], input_shape[0]))

    np_imgs_ref = np.array(imgs_ref)
    # if classify:
    #     option=OPTION[1]
    preds_ref = predict_imgs(network, np_imgs_ref, option=option)
    group_cls = group_from_files(files_ref)
    np_imgs_test = np.array(imgs_test)
    preds_test = predict_imgs(network, np_imgs_test, option=option)


    max_diff, min_diag, mat = cal_threshold(preds_ref, group_cls)
    abs_test_path = os.path.abspath(path_imgs_test[0])
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
            # fn = "%s_%s" % (path, os.path.basename(files_test[value]))
            fn=os.path.basename(files_test[value])
            shutil.copy(files_test[value], os.path.join(folder, fn))

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
        return OrderedDict(sorted(acc.items())), [correct, total]
    print(save_path)
    return 0


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

def compute_probs(files_test,imgs_test,network,input_shape,option=None):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class

    Returns
        probs : array of shape (m,m) containing distances

    '''

    # Compute all embeddings for all pics with current network
    np_imgs_test = np.array(imgs_test)
    embeddings = predict_imgs(network,np_imgs_test,option=option)
    m = embeddings.shape[0]
    nbevaluation = int(m * (m - 1) / 2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))
    group_cls_gt = group_from_files(files_test)
    # For each pics of our dataset
    k = 0
    for i in range(m):
        # Against all other images
        for j in range(i + 1, m):
            # compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
            if option==OPTION[1]:
                probs[k] = -compute_dist(embeddings[i, :], embeddings[j, :])
            else:
                probs[k]=get_similarity_score([embeddings[i, :]], [embeddings[j, :]])

            group_i=find_group_from_idx(group_cls_gt,i)
            group_j=find_group_from_idx(group_cls_gt,j)
            if group_i==group_j:
                y[k] = 1
            # print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
            else:
                y[k] = 0
            # print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
            k += 1
    return probs, y

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1], idx - 1
    else:
        return array[idx], idx


def draw_roc(fpr, tpr, thresholds,auc,init_epoch,save_path=None):
    # find threshold
    targetfpr = 1e-3
    _, idx = find_nearest(fpr, targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f}\nTPR : {1:.1%} \nThreshold={2}'.format(auc, recall, round(abs(threshold),5)))
    # show the plot
    fname="epoch-%s_roc.png"%init_epoch
    if save_path is not None:
        path=os.path.join(save_path,fname)
    else:
        path=fname
    plt.savefig(path)
    # plt.show()
    plt.clf()
    return abs(threshold)



def compute_interdist(network,files_ref,imgs_ref,option,input_shape):
    '''
    Computes sum of distances between all classes embeddings on our reference test image:
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings

    Returns:
        array of shape (nb_classes,nb_classes)
    '''
    # files_ref, imgs_ref = get_test_imgs(path_ref, size=(input_shape[1], input_shape[0]))

    np_imgs_ref = np.array(imgs_ref)
    group_cls = group_from_files(files_ref)
    # generates embeddings for reference images
    group_name=list(group_cls.keys())
    nb_group=len(group_name)
    res = np.zeros((nb_group, nb_group))
    ref_embeddings =predict_imgs(network, np_imgs_ref, option=option)

    for i in range(nb_group):
        group_i=group_name[i]
        embbeding_group_i=ref_embeddings[group_cls[group_i][0]]
        for j in range(nb_group):
            group_j=group_name[j]
            embbeding_group_j=ref_embeddings[group_cls[group_j][0]]
            res[i, j] = compute_dist(embbeding_group_i,embbeding_group_j)
    return res,group_name

def compute_metrics(probs, yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)
    #calculate accuracy

    return fpr, tpr, thresholds, auc


def draw_interdist(network,files_ref,imgs_ref,option,input_shape,nb_classes,init_epoch,save_path=None):
    interdist,group_name = compute_interdist(network,files_ref,imgs_ref,option=option,input_shape=input_shape)

    data = []
    for i in range(nb_classes):
        data.append(np.delete(interdist[i, :], [i]))

    fig, ax = plt.subplots()
    ax.set_title('Evaluating distance from each other after {0} epoch'.format(init_epoch))
    ax.set_ylim([0, 6])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data, showfliers=False, showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs, np.array(group_name))
    fname="epoch-%s_interdist.png"%init_epoch
    if save_path is not None:
        path=os.path.join(save_path,fname)
    else: path=fname
    plt.savefig(path)
    plt.close(fig)
    plt.clf()
    # plt.show()


def main_eval(network,path_imgs_test,path_imgs_ref,input_shape,nb_classes,epoch):
    files_test, imgs_test = get_test_imgs(path_imgs_test, size=(input_shape[1], input_shape[0]))
    files_ref, imgs_ref = get_test_imgs(path_imgs_ref, size=(input_shape[1], input_shape[0]))
    probs, yprob = compute_probs(files_test,imgs_test, network, input_shape, option=OPTION[1])
    fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
    threshold=draw_roc(fpr, tpr, thresholds,auc,epoch)
    draw_interdist(network, files_ref,imgs_ref, option=OPTION[1], input_shape=input_shape,nb_classes=nb_classes,init_epoch=epoch)

def all_eval(path_imgs_test, path_imgs_ref, network, nb_classes, input_shape,weight_path,save_dir):
    if eval:
        # vgg,dense,cnn-triplet,classify
        accuracy = []
        fieldnames = ["Group", "VGG origin", "Dense", "VGG with triplet", "Classify"]
        weight_name = os.path.basename(weight_path)
        data_name = os.path.basename(path_imgs_test)
        file_name = "%s_%s.csv" % (weight_name, data_name)
        full_path_file = os.path.join(save_dir, file_name)
        with open(full_path_file, "w+") as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(3):
                acc = test(path_imgs_test, path_imgs_ref, network, nb_classes, input_shape, save_dir, eval=eval, option=OPTION[i], classify=False)
                accuracy.append(acc)
            acc = test(path_imgs_test, path_imgs_ref, network, nb_classes, input_shape, save_dir, eval=eval, option=OPTION[1], classify=True)
            accuracy.append(acc)
            for key in accuracy[0][0]:
                group = key
                vgg_origin = accuracy[0][0][key]
                dense = accuracy[1][0][key]
                cnn_triplet = accuracy[2][0][key]
                classify = accuracy[3][0][key]
                row = {fieldnames[0]: group, fieldnames[1]: vgg_origin, fieldnames[2]: dense, fieldnames[3]: cnn_triplet, fieldnames[4]: classify}
                writer.writerow(row)
            final_row = {fieldnames[0]: "Accuracy", fieldnames[1]: accuracy[0][1], fieldnames[2]: accuracy[1][1], fieldnames[3]: accuracy[2][1], fieldnames[4]: accuracy[3][1]}
            writer.writerow(final_row)
            row_ = {fieldnames[0]: "", fieldnames[1]: round(accuracy[0][1][0]/accuracy[0][1][1],2), fieldnames[2]: round(accuracy[1][1][0]/accuracy[1][1][1],2),
                         fieldnames[3]: round(accuracy[2][1][0]/accuracy[2][1][1],2), fieldnames[4]: round(accuracy[3][1][0]/accuracy[3][1][1],2)}
            writer.writerow(row_)

if __name__=="__main__":
    new = 1
    if new:
        # weight_path = "../models/03_16_2020_0.0005/epoch-150_loss-0.00621.h5"
        # path_imgs_ref = "../data/ref_data_new"
        weight_path = "../models/03_23_2020_0.0005/epoch-102_loss-1.64523.h5"
        # weight_path = "../models/0309/epoch-86_loss-0.00906.h5"
        path_imgs_ref = ["/home/aimenext/cuongdx/ufj/data/ref"]
        # path_imgs_ref = "../data/ref_data_old"

    else:
        weight_path = "../models/0309/epoch-86_loss-0.00906.h5"
        path_imgs_ref = ["../data/ref_data_old"]

    bn=os.path.basename(weight_path)
    init_epoch=int(bn.split("_")[0].split("-")[1])
    nb_classes = 18
    input_shape = (512, 384, 3)
    thresh_cluster = 0.05
    thresh_classify = 0.4

    # test
    # path_imgs_test = "../data/data_test/new_group_0318"
    path_imgs_test = ["/home/aimenext/cuongdx/tripletloss/data/data_test/new_cate_2"]
    path_imgs_test=["/home/aimenext/cuongdx/ufj/data/all"]
    # main_cosin(path_imgs, weight_path,thresh=0.75,eval=False)

    # cal threshold
    # cosin_with_groundtruth(path_imgs_ref, weight_path)

    eval = False
    save_dir = "/home/aimenext/cuongdx/ufj/data/cluster"
    network = load_model(weight_path, nb_classes, input_shape)
    acc = test(path_imgs_test, path_imgs_ref, network, nb_classes, input_shape, save_dir, eval=eval, option=OPTION[0], classify=False,weight_path=weight_path,thresh_classify=thresh_classify,thresh_cluster=thresh_cluster)
    # main_eval(network,path_imgs_test,path_imgs_ref,input_shape,nb_classes,init_epoch)


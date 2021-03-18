from distutils.command.config import config

from network import Network
from dataset import DataGenerator
from keras.optimizers import Adam
import os
import argparse
from datetime import  datetime
import json
from keras import backend as K
from cluster import *
from keras.utils import multi_gpu_model


input_shape = (512, 384, 3)
batch_size = 8
epochs = 1000
DATA_PATH = ["/home/aimenext/cuongdx/tripletloss/data/train/03_16_2020","/home/aimenext/cuongdx/tripletloss/data/train/03_23_2020"]
WEIGHT_PATH="/home/aimenext/cuongdx/tripletloss/models/03_16_2020_0.0005/epoch-92_loss-0.01293.h5"
SAVED_PATH="/home/aimenext/cuongdx/tripletloss/models"
FOCUS=[str(i) for i in range(24,34)]
REFERENCE=["/home/aimenext/cuongdx/tripletloss/data/reference_data/0324"]
EVALUATION=["/home/aimenext/cuongdx/tripletloss/data/data_test/new_cate_2"]
GPUS=[0]

parser=argparse.ArgumentParser()
parser.add_argument("--batch_size",default=8,type=int)
parser.add_argument("--lr","--learning-rate",default=5e-4)
parser.add_argument("--weight_decay",default=1e-6)
parser.add_argument("--pretrained",type=str,default=WEIGHT_PATH)
parser.add_argument("--data",nargs="+",default=DATA_PATH)
parser.add_argument("--nb_features",help="number features of final layer",type=int,default=18)
parser.add_argument("--saved_path",type=str,default=SAVED_PATH)
parser.add_argument("--focus",nargs="+",default=FOCUS)
parser.add_argument("--eval",nargs="+",default=EVALUATION)
parser.add_argument("--reference",nargs="+",default=REFERENCE)
parser.add_argument("--gpus",nargs="+",default=GPUS)
parser.add_argument("--epoch",type=int,default=1000)
args=parser.parse_args()


def load_dataset(path_imgs):
    img_files = {}
    classes = []
    for path in path_imgs:
        for root, dir, files in os.walk(path):
            for file in files:
                cls = root.split(os.sep)[-1]
                full_path = os.path.join(root, file)
                if cls not in img_files:
                    img_files[cls] = [full_path]
                else:
                    img_files[cls].append(full_path)

                if cls not in classes:
                    classes.append(cls)
    return img_files, classes

def get_model_path(saved_path,lr):
    now = datetime.now()
    format_time = now.strftime("%m_%d_%Y")
    format_time = format_time + "_%s" % (lr)
    saved = os.path.join(saved_path, format_time)
    if not os.path.exists(saved):
        os.mkdir(saved)
    return saved

def save_info(dict_data):
    path_save=dict_data["saved"]
    with open(os.path.join(path_save,"config.json"),"w") as file:
        json.dump(dict_data,file)

def change_lr(current_lr,step_index):
    new_lr=current_lr*(0.8**step_index)
    return new_lr

def read_config(config_file):
    with open(config_file,"r+") as config:
        config_dict=json.load(config)
    return config_dict

def evaluate_model(files_eval,imgs_eval,files_ref,imgs_ref, network,epoch, nb_classes=18, input_shape=(512, 384, 3),save_path=""):
    np_imgs_ref = np.array(imgs_ref)

    preds_ref = predict_imgs(network, np_imgs_ref, option=OPTION[1])
    group_cls = group_from_files(files_ref)
    np_imgs_eval = np.array(imgs_eval)
    preds_test = predict_imgs(network, np_imgs_eval, option=OPTION[1])

    probs, yprob = compute_probs(files_eval,imgs_eval, network, input_shape, option=OPTION[1])
    fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
    threshold = draw_roc(fpr, tpr, thresholds, auc, epoch,save_path)
    if threshold<0.5:
        threshold=0.5
    draw_interdist(network,files_ref,imgs_ref, option=OPTION[1], input_shape=input_shape, nb_classes=nb_classes, init_epoch=epoch,save_path=save_path)
    result = test_classify(preds_test, preds_ref, group_cls, thresh=threshold)
    ref_group = group_from_files(files_ref)
    group_cls_gt = group_from_files(files_eval)
    acc, incorrect, detail = find_group_with_current_model(ref_group, group_cls_gt, result)
    print(incorrect)
    print(detail)
    json_data = result_to_json(detail, acc, files_eval,threshold)
    with open(os.path.join(save_path, "epoch-%s.json"%epoch), "w+") as json_file:
        json.dump(json_data, json_file)
    print(acc)
    correct = 0
    total = 0
    for k, v in acc.items():
        correct += v[0]
        total += v[1]
    print("correct {}, total {}".format(correct, total))
    return round(correct*100/total,3)

def main():
    data_path=args.data
    gpus=args.gpus
    if len(gpus)==1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        print("use single GPU to train model")

    nb_epochs=args.epoch
    imgs, classes = load_dataset(data_path)
    path_imgs_eval=args.eval
    path_imgs_ref=args.reference
    files_eval, imgs_eval = get_test_imgs(path_imgs_eval, size=(input_shape[1], input_shape[0]))
    files_ref, imgs_ref = get_test_imgs(path_imgs_ref, size=(input_shape[1], input_shape[0]))
    # nb_classes=len(classes)
    nb_classes=args.nb_features
    network = Network(nb_classes, input_shape)
    optimizer = Adam(lr=args.lr, decay=args.weight_decay)
    network_train = network.network_train
    parallel_model=None
    if len(gpus)>1:
        parallel_model=multi_gpu_model(network_train,gpus=gpus)
    else:
        print("using gpu {}---------------------------------------------------------------------".format(gpus[0]))
        parallel_model=network_train
    parallel_model.compile(loss=None, optimizer=optimizer)
    parallel_model.summary()
    # network_train.compile(loss=None, optimizer=optimizer)
    # network_train.summary()
    pretrained=args.pretrained
    # network_train.load_weights(pretrained)
    if pretrained is not None:
        parallel_model.load_weights(pretrained)
    batch_size=args.batch_size
    nb_hardest, nb_normal=int(batch_size/2),int(batch_size/2)
    saved_path=get_model_path(args.saved_path,args.lr)

    ##save info
    dict_info=vars(args)
    dict_info["input_shape"]=input_shape
    dict_info["nb_hardest"]=nb_hardest
    dict_info["nb_normal"]=nb_normal
    dict_info["saved"]=saved_path
    save_info(dict_info)



    ##init epoch
    ##format epoch-86_loss-0.00906.h5
    init_epoch=0
    height, width, channel = input_shape
    step_change_lr = 20
    focus_folder=args.focus
    save_after=1

    if args.pretrained is not None:
        bn=os.path.basename(args.pretrained)
        init_epoch=int(bn.split("_")[0].split("-")[1])
        print("init epoch: ",init_epoch)

    if init_epoch!=0:
        # current_lr=K.eval(network_train.optimizer.lr)
        current_lr=K.eval(parallel_model.optimizer.lr)
        step_index=init_epoch//step_change_lr
        new_lr = change_lr(current_lr, step_index)
        print("new lr: ", new_lr)
        # K.set_value(network_train.optimizer.lr, new_lr)
        K.set_value(parallel_model.optimizer.lr,new_lr)

    training_generator = DataGenerator(imgs, batch_size, width, height, channel, nb_hardest, nb_normal, network.base_network, nb_classes, classes, focus_folder=focus_folder,
                                       ratio_sample=4)

    epoch_end=init_epoch+nb_epochs
    for epoch in range(init_epoch+1,epoch_end):
        step_index=epoch//step_change_lr
        # current_lr=K.eval(network_train.optimizer.lr)
        current_lr=K.eval(parallel_model.optimizer.lr)
        if step_index>0 and epoch%step_change_lr==0:
            new_lr=change_lr(current_lr,step_index)
            print("new lr: ",new_lr)
            # K.set_value(network_train.optimizer.lr,new_lr)
            K.set_value(parallel_model.optimizer.lr,new_lr)
        if (training_generator.ratio_probs>1) and (epoch-init_epoch)%4==0:
            training_generator.ratio_probs-=1
        # network_train.fit(training_generator,epochs=10)
        total_loss=0
        for iter, batch_triplet in enumerate(training_generator):
            # loss = network_train.train_on_batch(batch_triplet, None)
            loss=parallel_model.train_on_batch(batch_triplet,None)
            total_loss+=loss
            print("epoch: {}, step: {}, loss: {:.5f},total loss: {:.5f} ".format(epoch, iter, loss,total_loss))
        if epoch % save_after == 0 and epoch>0:
            accuracy=evaluate_model(files_eval,imgs_eval,files_ref,imgs_ref,network,epoch,nb_classes,input_shape,saved_path)
            network_train.save_weights(
            "{}/epoch-{}_loss-{:.5f}_acc-{:.3f}.h5".format(saved_path,epoch, total_loss,accuracy))


if __name__ == "__main__":
    main()

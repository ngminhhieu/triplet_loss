from keras.applications import VGG16,ResNet50
import os
import glob
import numpy as np
import cv2
from draw_plot import plot_tsne
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import shutil
from sklearn.cluster import DBSCAN

input_shape=(1000,700,3)

def show_img(name,img):
	cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)
	cv2.imshow(name,img)
	cv2.waitKey(0)

def load_imgs(files):
	imgs=[]
	for file in files:
		img=cv2.imread(file)
		img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img=resize_img(img,shape=input_shape)
		img=normalizeMeanVariance(img)
		# show_img("img",img)
		print("max: ",np.max(img))
		imgs.append(img)

	return imgs

def resize_img(img,shape):
	img=cv2.resize(img,dsize=(shape[1],shape[0]))
	print("shape: ",img.shape)
	return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def elbow(model,X,k):
	visualizer=KElbowVisualizer(model,k=k)
	visualizer.fit(X)
	print("number of clusters: ",visualizer.elbow_value_)
	# visualizer.show()
	return visualizer.elbow_value_
#
def compare(values):
	results=[]
	for v in values:
		rs=values-v
		results.append(rs)
	return results





path="big_set"
files = glob.glob(os.path.join(path, "*"))
x_data=load_imgs(files)
x_data=np.array(x_data).reshape((-1,)+input_shape)
base_model=VGG16(weights="imagenet",include_top=False,input_shape=input_shape)
# base_model=ResNet50(weights="imagenet",include_top=False,input_shape=input_shape)
print(base_model.summary())
predicted=[]
for x_ in x_data:
	x_=np.expand_dims(x_,axis=0)
	pred=base_model.predict(x_)
	predicted.append(pred)
predicted=np.array(predicted)
print((-1,)+np.prod(predicted.shape[1:]))
predicted=predicted.reshape(-1,np.prod(predicted.shape[1:]))
print(predicted)
# plot_tsne(predicted,x_data,"out.png")



number_clusters=3
kmeans=KMeans()
# n_clusters=[elbow(kmeans,X=predicted,k=(3,20))]
nb_cls=10
n_clusters=[i for i in range(2,nb_cls)]
print(n_clusters)
for n_cluster in n_clusters:
	kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(predicted)
	targetdir="vgg_kmean_big_set"+os.sep+str(n_cluster)
	try:
		os.makedirs(targetdir)
	except OSError:
		pass
	# Copy with cluster name
	print("labels: ",set(kmeans.labels_))
	for i in set(kmeans.labels_):
		path=os.path.join(targetdir,str(i))
		if not os.path.exists(path):
			os.makedirs(path)
	for i, m in enumerate(kmeans.labels_):
		print("Copy: %s / %s" %(i, len(kmeans.labels_)))
		fn=os.path.basename(files[i])
		shutil.copy(files[i], os.path.join(os.path.join(targetdir,str(m)),fn))

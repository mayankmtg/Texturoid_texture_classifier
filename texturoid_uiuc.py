import sys
import os
import numpy as np
import cPickle as pickle
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from sklearn import svm
import cv2


def hog(img):
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bin_n = 16 # Number of bins
	bin = np.int32(bin_n*ang/(2*np.pi))

	bin_cells = []
	mag_cells = []

	cellx = celly = 8

	for i in range(0,img.shape[0]/celly):
		for j in range(0,img.shape[1]/cellx):
			bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
			mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])   

	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)
	hist = hist.astype("float")
	# transform to Hellinger kernel
	eps = 1e-7
	hist /= hist.sum() + eps
	hist = np.sqrt(hist)
	hist /= norm(hist) + eps

	return hist

def save(objects, name):
	with open(name, 'w') as fp:
		pickle.dump(objects, fp)



class HOG_LBP:
	def __init__(self, dataset_dir, load_saved=False):
		if(load_saved == True):
			print "Loading Dataset"
			self.dataset_dict = pickle.load(open('dataset_loaded_cv2.pck', 'r'))
		else:
			print "Reading Dataset"
			self.dataset_dict = {}
			for class_name in os.listdir(dataset_dir):
				self.dataset_dict[class_name] = []
				for filename in os.listdir(dataset_dir + class_name):
					if(".jpg" in filename):
						data_cv2 = cv2.imread(dataset_dir + "/" + class_name + "/" + filename, 0)
						# print data_cv2.shape
						self.dataset_dict[class_name].append(data_cv2)
			save(self.dataset_dict, "dataset_loaded_cv2.pck")

	def hog_feature_extract(self):
		winSize = (64,64)
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9
		derivAperture = 1
		winSigma = 4.
		histogramNormType = 0
		L2HysThreshold = 2.0000000000000001e-01
		gammaCorrection = 0
		nlevels = 64
		hog_extractor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
		hog_features = {}
		for class_name, objects in self.dataset_dict.iteritems():
			hog_features[class_name] = []
			for data_cv2 in objects:
				hog_features[class_name].append(hog_extractor.compute(data_cv2))
				# print hog_extractor.compute(data_cv2).shape
		return hog_features

	# def hog_feature_extract(self):
	# 	hog_features = {}
	# 	for class_name, objects in self.dataset_dict.iteritems():
	# 		hog_features[class_name] = []
	# 		for data_cv2 in objects:
	# 			hist = hog(data_cv2)
	# 			hog_features[class_name].append(hist)
	# 			# print hist.shape
	# 			# print hist
	# 			# break
	# 	return hog_features
	
	def lbp_feature_extract(self, radius=8, n_points=24, eps = 1e-7):
		# lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")
		lbp_features = {}
		for class_name, objects in self.dataset_dict.iteritems():
			lbp_features[class_name] = []
			for data_cv2 in objects:
				feature_lbp = local_binary_pattern(data_cv2,n_points,radius, method="default")
				(hist_lbp, _) = np.histogram(feature_lbp.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2))
				hist_lbp = hist_lbp.astype("float")
				hist_lbp /= (hist_lbp.sum() + eps)
				lbp_features[class_name].append(hist_lbp)
				# print 'm', hist_lbp.shape
				# print hist_lbp
		return lbp_features

def dict_to_array(my_dict):
	X = []
	Y = []
	ctr = 0
	for class_name, class_array in my_dict.iteritems():
		for i in class_array:
			X.append(i.ravel())
			Y.append(ctr)
		ctr+=1
	return np.array(X),np.array(Y)

def get_features_concat(my_dict1, my_dict2):
	X = []
	Y = []
	ctr = 0
	for class_name in my_dict1:
		for i in range(len(my_dict1[class_name])):
			X.append( np.concatenate((my_dict1[class_name][i], my_dict2[class_name][i])) )
			Y.append(ctr)
		ctr+=1
	print X[0].shape
	return np.array(X), np.array(Y)




load_dataset = True
load_hog = True
load_lbp = True

dataset_dir = "dataset_uiuc/"
hog_lbp = HOG_LBP(dataset_dir, load_dataset)

if(load_hog):
	print "Loading HOG Features"
	hog_features = pickle.load(open('hog_features.pck', 'r'))
else:
	print "Extracting HOG Features"
	hog_features = hog_lbp.hog_feature_extract()
	save(hog_features, "hog_features.pck")

if(load_lbp):
	print "Loading LBP Features"
	lbp_features = pickle.load(open('lbp_features.pck', 'r'))
else:
	print "Extracting LBP Features"
	lbp_features = hog_lbp.lbp_feature_extract()
	save(lbp_features, "lbp_features.pck")



X,Y = dict_to_array(hog_features)
X.shape, Y.shape
Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=.20, random_state=2)
clf = svm.LinearSVC()
clf.fit(Xtr, Ytr)
Y_pred = clf.predict(Xte)


print accuracy_score(Yte, Y_pred)


X,Y = dict_to_array(lbp_features)
Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=.25, random_state=2)
clf = svm.LinearSVC()
clf.fit(Xtr, Ytr)
Y_pred = clf.predict(Xte)

print accuracy_score(Yte, Y_pred)


X,Y = get_features_concat(hog_features, lbp_features)
Xtr,Xte,Ytr,Yte = train_test_split(X,Y, test_size=.25, random_state=2)
clf = svm.LinearSVC()
print Xtr
print Ytr
print X.shape, Y.shape
clf.fit(Xtr, Ytr)
Y_pred = clf.predict(Xte)

print accuracy_score(Yte, Y_pred)


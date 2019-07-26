'''
@David Xu Chang
edited last @25/7/2019
'''
import glob
import cv2
import numpy as np
import os.path as osp
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC

# Copied from original code by John
def print_data_stats(data,title='Data statistics'):
    total = 0
    for k,v in data.items():
        total += len(v)
    print('{}: loaded {} classes, a total of {} images.'.format(title, len(data), total))
# Copied from original code by John
class Dataset(object):
    def __init__(self, dataset_root):
        self.root = dataset_root
        self.train = None
        self.test = None
        self.classes = [] # string list
        self.num_classes = None

    def load_dataset(self):
        print('loading data from: {}'.format(self.root))
        if not osp.isdir(self.root):
            raise ValueError('Dataset root directory supplied not found.')
        self.train = self.load_subset(root=self.root,subset_type='train')
        self.test = self.load_subset(root=self.root,subset_type='test')
        self.num_classes = len(self.train)
        self.class_ids = list(range(len(self.classes)))
        self.class_name2id = dict(zip(self.classes, self.class_ids))
        self.class_id2name = dict(zip(self.class_ids, self.classes))
        print('data successfully loaded with stats:')
        print_data_stats(self.train,title='train')
        print_data_stats(self.test,title='test')
        
    def load_subset(self,root,subset_type):
        lib = {}
        class_dirs = glob.glob(str(self.root/subset_type/'*'))
        for c in class_dirs:
            c = Path(c)
            class_name = c.stem
            self.classes.append(class_name)
            image_list = glob.glob(str(c/'*.jpg'))
            lib[class_name] = image_list
            assert isinstance(image_list,list)
        return lib

data_path = Path.cwd().parent.parent / 'data'
data = Dataset(data_path)
data.load_dataset()
train = data.train
test = data.test
# print(test.get("Industrial")[65:70])
# train and test are two dict with key of categories and values of pathes
# print(len(train.get("Suburb"))) # Return <list>


# A func to find the ORB descriptor of an image

def calcOrbDes(img):
	orb = cv2.ORB_create()
	_, des = orb.detectAndCompute(img, None)
	return des

# A dict that contains all descriptor values	
desCollect = {}
All = {}
# descriptorSet = np.float32([]).reshape(0,32)
# A func to find all descriptors for training data
def initDescriptorSet():
	global descriptorSet
	global All
	for name, count in train.items():
		collection = train.get(name)
		coll = []
		descriptorSet = np.float32([]).reshape(0,32)
		for dir in collection:
			img = cv2.imread(dir, 0)
			des = calcOrbDes(img)
			coll.append(des)		
			descriptorSet = np.append(descriptorSet, des, axis=0)
		All[name] = coll
		desCollect[name] = descriptorSet

initDescriptorSet()
# A func to process the test data
def testDataProcessing():
	All_test = {}
	for name, count in test.items():
		collection = test.get(name)
		coll = []
		for dir in collection:
			img = cv2.imread(dir, 0)
			des = calcOrbDes(img)
			coll.append(des)
		All_test[name] = coll
	return All_test

All_test = testDataProcessing()
print(len(All_test.get("Suburb")))
# print(All_test)
bagSize = 200
# print(All) # All is check
# print(desCollect.get("Suburb").shape) # Testing for desCollect, seems good
# K-means algorithm to get the bags of words
allDescriptors = np.float32([]).reshape(0,32)
prediction = {}
# print(len(All.get("Suburb")))
def learnVocabPredict():
    global allDescriptors
    global prediction
    prediction_test = {}
    for name, count in desCollect.items():
		# print(type(desCollect.get(name))) checked <numpy.ndarray>
        allDescriptors = np.vstack((allDescriptors, desCollect.get(name)))
    kmeans = MiniBatchKMeans(n_clusters=bagSize).fit(allDescriptors)
    k_means_cluster = kmeans
    centers = kmeans.cluster_centers_
    print("the center has shape of ", centers.shape) # succesfully got an array
    np.save("center", centers) # save centers to local memory so that can be faster
    # So far I am correct
    # now prediction
    for name, count in All.items():
    	collection = All.get(name)
    	# print(collection)
    	predicted = np.float32([]).reshape(0,200)
    	for array in collection:
    		pre = kmeans.predict(array)
    		pre,_ = np.histogram(pre, bagSize)
    		# print(pre.size)
    		predicted = np.vstack((predicted, pre))
    	prediction[name] = predicted

    # print(All_test.get("Suburb")[0])
    # print(All.get("Suburb")[0])
    # print(type(All.get("Suburb")[0])==type(All_test.get("Suburb")[0]))
    prediction1 = {}
    print(len(All_test.get("Suburb")))
    for name, count in All_test.items():
    	collection1 = All_test.get(name)
    	count = 0
    	predicted1 = np.float32([]).reshape(0,200)
    	for col in collection1:
    		count += 1
    		# print(type(col))
    		if col is not None:
    			pred = kmeans.predict(col)
    			pred,_ = np.histogram(pred, bagSize)
    			predicted1 = np.vstack((predicted1, pred))
    	print(count)
    	prediction1[name] = predicted
    # 	# print(collection)
    # 	print(kmeans.predict(collection))
    print(prediction1.get("TallBuilding").size)
    # all the things seem to be multiplied by 2.15
    return prediction1
prediction1 = learnVocabPredict()

# def histForTest(a):
# 	pre_test = {}
# 	for name, count in All_test.items():
# 		collection = All_test.get(name)
# 		predicted = np.float32([]).reshape(0,200)
# 		print(len(collection))
# 		for col in collection:
# 			# print(col)
# 			col = col.reshape(-1,32)
# 			pre = a.predict(col)
# 			pre,_ = np.histogram(pre, bagSize)
# 			predicted = np.vstack((predicted, pre))
# 		pre_test[name] = predicted
# 	return pre_test
# prediction_test = histForTest(k__means)
# print(len(prediction.get("Suburb"))) # the length check passed!
# bagOfWords = np.load("center.npy") # load local center(bag of words)
# print(bagOfWords)
X = np.float32([]).reshape(0,200)
Y = np.float32([]).reshape(0,1)
# Now I start the support vector machine
def classifier():
	global X, Y
	for name, count in prediction.items():
		collection = prediction.get(name)
		for item in collection:
			X = np.vstack((X, item))
			Y = np.vstack((Y, name))
	if len(X) == len(Y):
		print("the size is matched")
	print("X.shape" , X.shape)
	print("Y.shape" , Y.shape)
	clf = SVC(kernel='poly')
	supVecMach = clf.fit(X, Y)
	print("the training accuracy is", clf.score(X, Y))
	x = np.float32([]).reshape(0,200)
	y = np.float32([]).reshape(0,1)
	for name, count in prediction1.items():
		collection = prediction1.get(name)
		for item in collection:
			x = np.vstack((x, item))
			y = np.vstack((y, name))
	print(x.size, y.size, X.size, Y.size)
	return clf.score(x, y)
	# return supVecMach
supVecM = classifier()
print(supVecM)

# def tester(a):


# print(tester(supVecM))




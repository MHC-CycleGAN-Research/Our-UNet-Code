from defines import *
from model import *
from data import *
from datetime import datetime

import cv2
import numpy as np
import glob
import os

def mergeIm():
	list = os.listdir(os.path.join(PARAM_PATH_TEST,PARAM_IMG_FOLDER))
	number_files = len(list)
	
	ext = ".png"
	
	# Paths for test, result, label images
	path_test = os.path.join(os.path.join(PARAM_PATH_TEST,PARAM_IMG_FOLDER), "*" + ext)
	path_result = os.path.join(PARAM_PATH_TEST_RESULTS, "*" + ext)
	path_label = os.path.join(os.path.join(PARAM_PATH_TEST,PARAM_MSK_FOLDER), "*" + ext)
	
	# Information of images
	images_test = [cv2.imread(img) for img in glob.glob(path_test)]
	images_result = [cv2.imread(img) for img in glob.glob(path_result)]
	images_label = [cv2.imread(img) for img in glob.glob(path_label)]
	
	h,w,d = images_test[0].shape
	
	height = h * number_files
	width = w * 3
	output = np.zeros((height,width,3))
	
	# current row
	n = 0
	for i in number_files:
		# test image | result image | ground truth
    		output[n:n+h,0:w] = images_test[i]
		output[n:n+h,w:w*2] = images_result[i]
		output[n:n+h,w*2:w*3] = images_label[i]
		n += h
	
	cv2.imwrite("output.png", output)
	

if __name__ == '__main__':

	# step0: enable GPU version
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	if PARAM_ACTION == 1:
		# step1: create training set
		myGene = trainGenerator(PARAM_BATCHES, 
					PARAM_PATH_TRAIN, 
					PARAM_IMG_FOLDER, 
					PARAM_MSK_FOLDER, 
					PARAM_DATA_ARGS, 
					save_to_dir = PARAM_AUG_FOLDER)
		
		hdf5 = '.hdf5'
		PARAM_SAVED_MODEL = datetime.now().strftime("%Y%m%d-%H%M%S") + hdf5

		# setp2: set up unet model
		model = unet()

		# step3: set up model checkpoint save path
		model_checkpoint = ModelCheckpoint( PARAM_SAVED_MODEL, 
						    monitor = PARAM_METRICS, 
						    verbose = 1, 
						    save_best_only = PARAM_SAVE_BEST_ONLY)

		# step4: start training the model
		model.fit_generator(myGene,	
				    steps_per_epoch = PARAM_EPOCH_STEPS,
				    epochs = PARAM_N_EPOCHS,
				    callbacks = [model_checkpoint])

	elif PARAM_ACTION == 2:

		# setp1: load trained model and weights
		model = unet()
		model.load_weights(PARAM_SAVED_MODEL)   

		# step2: create testing set
		testGeneX, testGeneY = testGenerator(PARAM_PATH_TEST,
						     PARAM_IMG_FOLDER, 
						     PARAM_MSK_FOLDER)

		# step3: evaluate model performance
		results = model.predict(testGeneX, PARAM_N_TESTS, verbose=1)

		# step4: save results
		np.save(PARAM_PATH_TEST_NPY, results)
		saveResult(PARAM_PATH_TEST_RESULTS,results)

		# TODO: visualization and analysis (Dice IoU)
		mergeIm()

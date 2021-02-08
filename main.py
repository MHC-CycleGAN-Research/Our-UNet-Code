from defines import *
from model import *
from data import *
	

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

		PARAM_ACTION = 2

	if PARAM_ACTION == 2:

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

		# step5: visualization and dice/IoU?
		mergeIm(PARAM_PATH_TEST, PARAM_IMG_FOLDER, PARAM_MSK_FOLDER, 
				PARAM_PATH_TEST_RESULTS, PARAM_PATH_TEST_ALL_IMG)

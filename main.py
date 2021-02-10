from defines import *
from model import *
from data import *
import sys


if __name__ == '__main__':
	# step0: enable GPU version
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	if len(sys.argv) > 1:

		PARAM_CP = int(sys.argv[1])

		if PARAM_CP == 1:
			print("RUN_TIME_ARGUEMENT = (cartesian) "+str(sys.argv[1]))
		elif PARAM_CP == 2:
			print("RUN_TIME_ARGUEMENT = (polar) "+str(sys.argv[1]))
		else:
			input("Invalid RUN_TIME_ARGUEMENT detected. you received: "+str(sys.argv[1]))
	else:

		input("No RUN_TIME_ARGUEMENT detected.. something's wrong?")


	if PARAM_ACTION == 1:

		# step1: create training set
		if PARAM_CP ==1:
			myGene = trainGenerator(PARAM_BATCHES, 
						PARAM_PATH_TRAIN_C, 
						PARAM_IMG_FOLDER, 
						PARAM_MSK_FOLDER, 
						PARAM_DATA_ARGS, 
						save_to_dir = PARAM_AUG_FOLDER_C)

		if PARAM_CP ==2:
			myGene = trainGenerator(PARAM_BATCHES, 
						PARAM_PATH_TRAIN_P, 
						PARAM_IMG_FOLDER, 
						PARAM_MSK_FOLDER, 
						PARAM_DATA_ARGS, 
						save_to_dir = PARAM_AUG_FOLDER_P)

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
		if PARAM_CP ==1:
			testGeneX, testGeneY = testGenerator(PARAM_PATH_TEST_C,
							     PARAM_IMG_FOLDER, 
							     PARAM_MSK_FOLDER)

			# step3: evaluate model performance
			results = model.predict(testGeneX, PARAM_N_TESTS, verbose=1)

			# step4: save results
			np.save(PARAM_PATH_TEST_NPY_C, results)
			saveResult(PARAM_PATH_TEST_RESULTS_C,results)

			# step5: visualization and dice/IoU?
			mergeIm(PARAM_PATH_TEST_C, PARAM_IMG_FOLDER, PARAM_MSK_FOLDER, 
				PARAM_PATH_TEST_RESULTS_C, PARAM_PATH_TEST_ALL_IMG_C)
		if PARAM_CP ==2:
			testGeneX, testGeneY = testGenerator(PARAM_PATH_TEST_P,
							     PARAM_IMG_FOLDER, 
							     PARAM_MSK_FOLDER)

			# step3: evaluate model performance
			results = model.predict(testGeneX, PARAM_N_TESTS, verbose=1)

			# step4: save results
			np.save(PARAM_PATH_TEST_NPY_P, results)
			saveResult(PARAM_PATH_TEST_RESULTS_P,results)

			# step5: visualization and dice/IoU?
			mergeIm(PARAM_PATH_TEST_P, PARAM_IMG_FOLDER, PARAM_MSK_FOLDER, 
				PARAM_PATH_TEST_RESULTS_P, PARAM_PATH_TEST_ALL_IMG_P)

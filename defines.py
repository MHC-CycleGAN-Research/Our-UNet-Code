from data import add_noise

PARAM_ACTION = 1 							# 1 for training, 2 for testing

PARAM_BATCHES = 2
PARAM_N_EPOCHS = 10
PARAM_N_TESTS = 100
PARAM_EPOCH_STEPS = 500

PARAM_SAVE_BEST_ONLY = True

PARAM_PATH_TRAIN = './data/endoscopic/train'
PARAM_AUG_FOLDER = './data/endoscopic/train/aug'
PARAM_PATH_TEST =  './data/endoscopic/test'
PARAM_PATH_TEST_RESULTS = './data/endoscopic/test/predict'
PARAM_PATH_TEST_NPY = './data/endoscopic/test/imgs_endoscopic.npy'

PARAM_SAVED_MODEL = 'unet_endoscopic.hdf5'    # TODO: log current system time as file name
PARAM_IMG_FOLDER = 'image'
PARAM_MSK_FOLDER = 'label'
PARAM_METRICS = 'loss'						# TODO: motitor more metrics... look up the options.

PARAM_DATA_ARGS = dict(rotation_range = 	190,		# 0.2 			# TODO: improve the data augmentation
                width_shift_range =			0.0,		# 0.05	
                height_shift_range =		0.0,		# 0.05
                shear_range	= 				0.35,		# 0.05
                zoom_range = 				0.0,		# 0.05
                horizontal_flip = 			True,		# True
                fill_mode = 				'nearest',	# 'nearest'
                preprocessing_function =	None,  # None
                rescale =                   1./255)     # None




from datetime import datetime

PARAM_ACTION = 1 							# 1 for training, 2 for testing

PARAM_BATCHES = 2
PARAM_N_EPOCHS = 50
PARAM_N_TESTS = 100
PARAM_EPOCH_STEPS = 100

PARAM_SAVE_BEST_ONLY = True

PARAM_SYSTEM_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
PARAM_PATH_TRAIN = './data/endoscopic/train'
PARAM_AUG_FOLDER = '/aug_' + PARAM_SYSTEM_TIME
PARAM_PATH_TEST =  './data/endoscopic/test'
PARAM_PATH_TEST_RESULTS = './data/endoscopic/test/predict_' + PARAM_SYSTEM_TIME
PARAM_PATH_TEST_NPY = './data/endoscopic/test/predict_' + PARAM_SYSTEM_TIME + '.npy'
PARAM_PATH_TEST_ALL_IMG = './data/endoscopic/test/predict_' + PARAM_SYSTEM_TIME + '.png'
PARAM_SAVED_MODEL = 'unet_' + PARAM_SYSTEM_TIME + '.hdf5'

PARAM_IMG_FOLDER = 'image'
PARAM_MSK_FOLDER = 'label'
PARAM_METRICS = 'loss'								# TODO: motitor more metrics... look up the options.

PARAM_DATA_ARGS = dict(rotation_range = 0.2,      	# TODO: improve the data augmentation
                width_shift_range =		0.05,
                height_shift_range =	0.05,
                shear_range	= 			0.05,
                zoom_range = 			0.05,
                horizontal_flip = 		True,
                fill_mode = 			'nearest')




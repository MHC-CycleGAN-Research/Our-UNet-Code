from helper_functions import *
from data import *

n_test = 956

data_gen_args = dict(rotation_range=50,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.35,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    )
                    
myGene = trainGenerator(2,'data/endoscopic/train','image','label',data_gen_args,save_to_dir = 'data/endoscopic/train/aug')
model = UNetPlusPlus(256, 256, 3)
model_checkpoint = ModelCheckpoint('unet_endoscopic.hdf5', monitor = 'loss', verbose=1, save_best_only=True)

model.compile('Adam', 'binary_crossentropy', ['accuracy', mean_iou])

model.fit_generator(myGene,steps_per_epoch=500,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data/endoscopic/test/image", num_image = n_test, flag_multi_class = True, as_gray = False)
results = model.predict_generator(testGene,n_test,verbose=1)

np.save('./results/imgs_endoscopic_polar.npy', results)

saveResult("data/endoscopic/test/predict",results)


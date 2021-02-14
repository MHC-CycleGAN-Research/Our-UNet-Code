from model import *
from data import *
from tensorflow.keras import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

n_test = 100

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    )
                    
myGene = trainGenerator(2,'data/endoscopic/train','image','label',data_gen_args,save_to_dir = 'data/endoscopic/train/aug')
model = unet()

model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = ['accuracy', dice_coef_loss])

model_checkpoint = ModelCheckpoint('unet_endoscopic.hdf5', monitor = 'loss', verbose=1, save_best_only=True)

model.fit_generator(myGene,steps_per_epoch=150,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data/endoscopic/test/image", num_image = n_test, flag_multi_class = True, as_gray = False)
results = model.predict_generator(testGene,n_test,verbose=1)

np.save('./results/imgs_endoscopic_polar.npy', results)

saveResult("data/endoscopic/test/predict",results)


from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_test = 10

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/endoscopic/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_endoscopic.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=30,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data/endoscopic/test", num_image = n_test, flag_multi_class = True, as_gray = False)
results = model.predict_generator(testGene,n_test,verbose=1)

np.save('./results/imgs_endoscopic.npy', results)

saveResult("data/endoscopic/test",results)

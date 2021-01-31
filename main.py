from model import *
from data import *
import numpy as np
import tensorflow as tf
import random
import cv2
from PIL import Image
from pathlib import Path
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Takes as input path to image file and returns 
# resized 3 channel RGB image of as numpy array of size (256, 256, 3)
'''
def getPic(img_path):
    return np.array(Image.open(img_path).convert('RGB').resize((256,256),Image.ANTIALIAS))

# Return the images and corresponding labels as numpy arrays
def get_ds(data_path, label_path):
    img_paths = list()
    lbl_paths = list()
    
    # Recursively find all the image files from the path data_path
    for img_path in glob.glob(data_path+"/*"):#"/**/*"):
        img_paths.append(img_path)
    
    # Recursively find all the image files from the path label_path
    for lbl_path in glob.glob(label_path+"/*"):
        lbl_paths.append(lbl_path)
        
    images = np.zeros((len(img_paths),256,256,3))
    labels = np.zeros((len(lbl_paths),256,256,3))

    # print how many images found meow
    print(len(img_paths))
      
    # Read and resize the images
    # Get the encoded labels
    for i, img_path in enumerate(img_paths):
        images[i] = getPic(img_path)
              
    for i, lbl_path in enumerate(lbl_paths):
        labels[i] = getPic(lbl_path)
        
    return images,labels

train_X, train_y = get_ds("./data/membrane/train/image/", "./data/membrane/train/label/")
test_X, test_y = get_ds("./data/membrane/test/", "./data/membrane/test/")
'''
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)

'''
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

model.fit(train_X,train_y,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

results = model.predict(testX)

saveResult("data/membrane/test",results)
'''
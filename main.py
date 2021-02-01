from model import *
from data import *
import numpy as np
import tensorflow as tf
import random
import sys
import string
rospath = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if str(sys.path).find(rospath) != -1:
    sys.path.remove(rospath) # in order to import cv2 under python3
    print('ROS path temporarily removed.')
import cv2
from PIL import Image
from pathlib import Path
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print('\n\n================ Welcome to Our UNet ================')
mychoice = input("\n\nChoose 1 for Membrane example, 2 for Endoscopic images.")
myaction = input("\n\nChoose 1 for Training, 2 for Testing.")


if mychoice == '1':
    if myaction == '1':
        data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
        myGene = trainGenerator(2,'./data/membrane/train','image','label',data_gen_args,save_to_dir = None)

        model = unet()
        model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
        model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])

        print('\n\nDone training!')
        myaction = input("\n\nChoose 1 for Exit, 2 for Testing.")

    if myaction == '2':
        testGene = testGenerator("./data/membrane/test")
        model = unet()
        model.load_weights("unet_membrane.hdf5")
        results = model.predict_generator(testGene,30,verbose=1)
        saveResult("./data/membrane/test",results)

    else:
        print("Invalid action. Goodbye!")

elif mychoice == '2':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    

    # Return the images and corresponding labels as numpy arrays
    def get_ds(data_path, label_path):
        img_paths = list()
        lbl_paths = list()
        
        # Recursively find all the image files from the path data_path
        for img_path in glob.glob(data_path+"/*"):
            img_paths.append(img_path)
        
        # Recursively find all the image files from the path label_path
        for lbl_path in glob.glob(label_path+"/*"):
            lbl_paths.append(lbl_path)
            
        images = np.zeros((len(img_paths),256,256,3))
        labels = np.zeros((len(lbl_paths),256,256,1))
          
        # Read and resize the images
        # Get the encoded labels
        for i, img_path in enumerate(img_paths):
            # Takes as input path to image file and returns 
            # resized 3 channel RGB image of as numpy array of size (256, 256, 3)
            images[i] = np.array(Image.open(img_path).convert('RGB').resize((256,256),Image.ANTIALIAS))
                  
        for i, lbl_path in enumerate(lbl_paths):
            labels[i] = np.array(Image.open(lbl_path).convert('L').resize((256,256),Image.ANTIALIAS)).reshape((256,256,1))
        
        input(str(len(img_paths))+'   '+str(len(lbl_paths)))
        return images,labels

    if myaction == '1':
        train_X, train_y = get_ds("./data/endoscopic/train/image/", "./data/endoscopic/train/label/")
        model = unet(pretrained_weights = None, input_size = (256,256,3))
        model_checkpoint = ModelCheckpoint('unet_endoscopic.hdf5', monitor='loss',verbose=1, save_best_only=True)
        model.fit(train_X,train_y,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])

        print('\n\nDone training!')
        myaction = input("\n\nChoose 1 for Exit, 2 for Testing.")

    if myaction == '2':        
        test_X, test_y = get_ds("./data/endoscopic/test/", "./data/endoscopic/test/")
        model = unet(pretrained_weights = None, input_size = (256,256,3))
        model.load_weights("unet_endoscopic.hdf5")      
        results = model.predict(test_X)

        saveResult("./data/endoscopic/test",results)

    else:
        print("Invalid action. Goodbye!")

else:
    print("Invalid option. Goodbye!")
import cv2 #import opencv
import numpy as np
import matplotlib.pyplot as plt #import pyplot and alias it as plt
import os #to work with directories
import random as rnd
from patchify import patchify, unpatchify
from pathlib import Path #help with path mess
import tensorflow as tf
from keras.utils import normalize
from U_net import u_net_model


#Load data
cur_dir= Path(__file__).parent
path1= Path('U-net_staffline/Training_BW') #attention, this / doesnt work!! you have to use this \
path2= Path('U-net_staffline/Training_GR')
path3= Path('U-net_staffline/Training_GT')

# Read the images folder like a list
image_BW = os.listdir(cur_dir/path1)
image_GR = os.listdir(cur_dir/path2)
image_GT = os.listdir(cur_dir/path3)

#Read image folder like a list
images_GT = []
images_GR = []
images_BW = []

for file in image_BW:
    images_BW.append(file)
for file in image_GR:
    images_GR.append(file)
for file in image_GT:
    images_GT.append(file)
    
#Visualize example  
""" 
l= rnd.randint(1,len(images_BW))

plt.subplot(1,2,1)
img_vis_BW= cv2.imread(str(path1/images_BW[l]),cv2.IMREAD_GRAYSCALE)
plt.imshow(img_vis_BW,'gray')
plt.subplot(1,2,2)
img_vis_GT= cv2.imread(str(path2/images_GT[l]),cv2.IMREAD_GRAYSCALE)
plt.imshow(img_vis_GT,'gray')
plt.show()
cv2.waitKey(0)   
"""    
    
#Patchify images and write them to new folder
def def_patchify_img(image_set,target_dir,height,width,step_,img_number,path): #!! make sure to corectly indent, ctrl + ] + (+/-) for indentation of multiple lines
    """
    Function for making patches of large images. As an input we need array of image names, target directory for new 
    images height and width of patch. Then also step for patch and number of images from image_set to be patched
    
    By adjusting the patch size to have a similar aspect ratio to the original image, the neural network can 
    better capture the relevant features of the image and produce a more accurate segmentation map.
    A patch size of 256x256 or 512x512 is commonly used in U-Net applications.
    Choose an overlap: Overlapping patches can help avoid artifacts at the patch boundaries and improve the quality 
    of the segmentation map. A 50% overlap is commonly used, but the amount of overlap can be adjusted based on the 
    specific requirements of your application.
    Data augmentation: To improve the robustness of the U-Net model, data augmentation can be applied to the patches. 
    Common data augmentation techniques include random rotation, scaling, flipping, and adding noise.
    """
    #Path.mkdir(cur_dir/target_dir) #create new folder
    
    p=1
    for k in range(img_number):
        img_to_patch= cv2.imread(str(path/image_set[k]),cv2.IMREAD_GRAYSCALE) #from image_set choose one image to patch
        patches_img = patchify(img_to_patch,(height,width), step = step_) #create patches of choosen img
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch = patches_img[i,j,:,:]  #extract each patch from img
                cv2.imwrite(str(Path(target_dir)/'img') + '_'+str(p) + '.png', single_patch)   #save each patch to img_xx.png file in desired folder
                p= p+1
                
   
#def_patchify_img(images_BW,'U-net_staffline\Training_BW_patches',256,256,256,100,path1) 
#def_patchify_img(images_GR,'U-net_staffline\Training_GR_patches',256,256,256,100,path2)  
#def_patchify_img(images_GT,'U-net_staffline\Training_GT_patches',256,256,256,100,path3)     
               

#Create tensor of original patches
image_dataset=[]
images_path= Path('U-net_staffline/Training_GR_patches')
images= os.listdir(images_path)
for i in range(len(images)):
    image_dataset.append(cv2.imread(str(images_path/images[i]),cv2.IMREAD_GRAYSCALE))
    
#Create tensor of mask patches
mask_dataset=[]
masks_path= Path('U-net_staffline/Training_GT_patches')
masks= os.listdir(masks_path)
for i in range(len(masks)):
    mask_dataset.append(cv2.imread(str(masks_path/masks[i]),cv2.IMREAD_GRAYSCALE))    

image_dataset = np.expand_dims(normalize(np.array(image_dataset)),3) #convert to numpy array and normalize pixels to values 0-1 (float 64), expand dimension so it is 7845,256,256,1
mask_dataset = np.expand_dims(np.array(mask_dataset),3)/255 #np expand dims to add 4th dimension (we have only grey img)




from sklearn.model_selection import train_test_split
# Use scikit-learn's function to split the dataset
# Here, I have used 20% data as test/valid set
X_train, X_valid, y_train, y_valid = train_test_split(image_dataset, mask_dataset, test_size=0.2, random_state=0) #for random state we can use 0(uses numpy random) or define seed number for number generator

#Check that patch of original and groundtruth match
image_number = rnd.randint(0, len(X_train))

#Sanity check that input and output(mask) match
fig, axis = plt.subplots(1,2,figsize=(20,5))
axis[0].imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
axis[0].set_title('Input img')
axis[1].imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
axis[1].set_title('Groundtruth(mask)')
fig.suptitle('Sanity check')
plt.show()

#Create U-net model
u_net= u_net_model((256,256,1),16,2) #u_net_model(input_size,nfilters,classes)
u_net.summary()

#Compile model, set minibatches and epochs
u_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
results= u_net.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(X_valid,y_valid))

#Plot 
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,5))
axis1.plt(results.history['loss'],color='red',label='training loss')
axis1.plt(results.history['val_loss'],color='blue',label='validation loss')
axis1.set_title('Loss comparison')
axis2.plt(results.history['accuracy'],color='red',label='accuracy')
axis2.plt(results.history['val_accuracy'],color='blue',label='validation accuracy')
axis2.set_title('Accuracy comparison')
fig.legend()
fig.title('Training of U-net')
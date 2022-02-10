import os
from PIL import Image
import pandas as pd
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt


def min_size(path_of_image):
    """
    Checking min size available in all folder.
    This will help you to give idea, which size you have to select for resizing.
    
    """
    # wlking across directory to open all available images
    size_images = {}
    for dirpath, _, filenames in os.walk(path_of_image):
        for path_image in filenames:
            image = os.path.abspath(os.path.join(dirpath, path_image))
            
            # checking images size and storing in a dict to compare
            with Image.open(image) as img:
                width, heigth = img.size
                size_images[path_image] = {'width': width, 'heigth': heigth}
                
    # Creating a small DF to check min & max size of images
    df_image_desc = pd.DataFrame(size_images).T
    min_width = df_image_desc['width'].min()
    min_height = df_image_desc['heigth'].min()
    
    return min_height, min_width

def load_image_from_folder(path, basewidth, hsize):
    
    """
    Loading all images in a numpy array with labels
    
    """
    # creating temp array
    image_array = []
    labels = []
    # directory walking started
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file != []:
                # trying to get path of each images
                path_updated = os.path.join(subdir, file)
                # fetching lables from directory names
                label = subdir.split("/")[-1]
                labels.append(label)
                # Converting image & resizing it
                img = Image.open(path_updated).convert('L')
                img = img.resize((basewidth, hsize), Image.ANTIALIAS)
                frame = asarray(img)
                # appending array of image in temp array
                image_array.append(frame)
                
    # Now i have to convert this images to array channel format which can be done using zero matrix
    # creating a dummy zero matrix of same shape with single channel
    
    image_array1 = np.zeros(shape=(np.array(image_array).shape[0], hsize, basewidth,  1))
    for i in range(np.array(image_array).shape[0]):
        # finally each sub matrix will be replaced with respective images array
        image_array1[i, :, :, 0] = image_array[i]
    
    return image_array1, np.array(labels)

def vis_training(hlist, start=1):
    
    """
    This function will help to visualize the loss, val_loss, accuracy etc.
    
    """
    # getting history of all kpi for each epochs
    loss = np.concatenate([hlist.history['loss']])
    val_loss = np.concatenate([hlist.history['val_loss']])
    acc = np.concatenate([hlist.history['accuracy']])
    val_acc = np.concatenate([hlist.history['val_accuracy']])
    epoch_range = range(1,len(loss)+1)
    
    # Block for training vs validation loss
    plt.figure(figsize=[12,6])
    plt.subplot(1,2,1)
    plt.plot(epoch_range[start-1:], loss[start-1:], label='Training Loss')
    plt.plot(epoch_range[start-1:], val_loss[start-1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # Block for training vs validation accuracy
    plt.subplot(1,2,2)
    plt.plot(epoch_range[start-1:], acc[start-1:], label='Training Accuracy')
    plt.plot(epoch_range[start-1:], val_acc[start-1:], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
datagen = ImageDataGenerator(        
        rotation_range = 90,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True)
import numpy as np
import os
from PIL import Image
image_directory = r'./Training/malignant/'
SIZE = 112
dataset = []
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):    
    if (image_name.split('.')[1] == 'jpg'):        
        image = io.imread(image_directory + image_name)        
        image = Image.fromarray(image, 'RGB')        
        dataset.append(np.array(image))
x = np.array(dataset)
i = 0
for batch in datagen.flow(x, batch_size=5,
                          save_to_dir= r'./Training/malignant',
                          save_prefix='dr',
                          save_format='jpg'):    
    i += 1    
    if i > len(my_images):        
        break

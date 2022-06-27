import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import _pickle as cPickle
#from glob import glob
#import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras import backend as K 


################################################################################
#Loading images functions
################################################################################
def load_images(load_dir, image_list):
    lst = [np.asarray(Image.open(load_dir + '/' + filename + '.jpg').convert('RGB')) for filename in image_list]
    return lst

def load_prepare_dataset(dataset_csv, folder):

    malignant = train_data.loc[(train_data['MEL'] == 1)| (train_data['BCC'] == 1)]
    benign = train_data.loc[(train_data['MEL'] == 0) & (train_data['BCC'] == 0)]
    
    # Load in training pictures 
    benign_images = load_images(folder, benign['image'])
    malignant_images = load_images(folder, malignant['image'])
    print(f'len malignant:{len(malignant)}')
    print(f'len benign:{len(benign)}')
    X_benign = np.array(benign_images, dtype='uint8')
    X_malignant = np.array(malignant_images, dtype='uint8')

    #####################################
    #data augmentation
    #####################################

    '''
        img_generator = ImageDataGenerator(horizontal_flip=True)
    img_generator = img_generator.flow(malignant_images, batch_size=len(malignant_images))
    batch = img_generator.next()

    for augm_image in batch:
        X_malignant.append(np.array(augm_image, dtype='uint8'))
    
    img_generator = ImageDataGenerator(vertical_flip=True)
    img_generator = img_generator.flow(malignant_images, batch_size=len(malignant_images))
    batch = img_generator.next()

    for augm_image in batch:
        X_malignant.append(np.array(augm_image, dtype='uint8'))
    '''

    print('After augmentation')
    print(f'len malignant:{X_malignant.shape[0]}')
    print(f'len benign:{X_benign.shape[0]}')

    # Create labels
    y_benign = np.zeros(X_benign.shape[0])
    y_malignant = np.ones(X_malignant.shape[0])

    # Merge data 
    x = np.concatenate((X_benign, X_malignant), axis = 0)
    y = np.concatenate((y_benign, y_malignant), axis = 0)

    # Shuffle data
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]

    y = to_categorical(y, num_classes= 2)
    x = x/255.

    return x, y
################################################################################


################################################################################
#Model building functions
################################################################################
def run_model(model):
    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs= epochs, batch_size= batch_size, verbose=1, 
                        callbacks=[learning_rate_reduction]
                    )
                    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def build_model(input_shape= (224,224,3), lr = 1e-3, num_classes= 2, init= 'normal', activ= 'relu'):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),padding = 'Same',input_shape=input_shape,
                     activation= activ, kernel_initializer='glorot_uniform', strides =(2,2)))
    model.add(MaxPool2D(pool_size = (3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3),padding = 'Same', 
                     activation =activ, kernel_initializer = 'glorot_uniform', strides =(2,2)))
    model.add(MaxPool2D(pool_size = (3, 3)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=init))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    optimizer = Adam(lr=lr)

    model.compile(optimizer = optimizer ,loss = "binary_crossentropy", metrics=["AUC"])
    return model


def save_model(model, model_filename):
    model_json = model.to_json()

    with open(model_filename + ".json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights(model_filename + ".h5")
    print("Saved")

    del model
    K.clear_session()
################################################################################



if __name__ == '__main__':
    choice = int(input('Select task; \n 1) Train CNN 2) Train ResNet50 3) Train VGG16'))

    #Load dataset csv and create dataset vector and label
    train_folder = './Training'
    train_data = pd.read_csv(train_folder + '/' + 'ISIC2018_Training_GroundTruth.csv')

    X, y = load_prepare_dataset(train_data, train_folder)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    #NN parameters
    input_shape = (224,224,3)
    lr = 1e-4
    init = 'normal'
    activ = 'relu'
    epochs = 50
    batch_size = 64

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_auc', patience=5, verbose=1, factor=0.5, min_lr=1e-7)

    if choice == 1:
        #CNN cross-validation
        kfold = KFold(n_splits=3, shuffle=True, random_state=17)

        cvscores = []
        for train, test in kfold.split(X_train, y_train):
            model = build_model(lr=lr, 
                        init= init, 
                        activ= activ, 
                        input_shape= input_shape)
            
            model.fit(X_train[train], y_train[train], epochs=epochs, batch_size=batch_size, verbose=0)
            scores = model.evaluate(X_train[test], y_train[test], verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            K.clear_session()
            del model
            
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

        #CNN testing
        model = build_model(lr=lr, init= init, activ= activ, input_shape= input_shape)

        model.fit(X_train, y_train, epochs=epochs, batch_size= batch_size, verbose=1, callbacks=[learning_rate_reduction])

        y_pred = model.predict_classes(X_test)

        print(accuracy_score(np.argmax(y_test, axis=1),y_pred))

        save_model(model, 'cnn_model')
    
    elif choice == 2 :

        #ResNet50 parameters
        input_shape = (224,224,3)
        lr = 1e-5
        epochs = 10
        batch_size = 64

        #Building ResNet50
        resnet50_model = ResNet50(include_top=False, weights= 'imagenet', input_tensor=None, input_shape=input_shape, pooling='avg', classes=2)

        for layer in resnet50_model.layers:
            layer.trainable = False
            
        model = Sequential()
        model.add(resnet50_model)
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))


        model.compile(optimizer = 'Adam' ,loss = "binary_crossentropy", metrics=["accuracy"])

        run_model(model)

        #Testing ResNet50
        model.fit(X_train, y_train,epochs=epochs, batch_size= epochs, verbose=1, callbacks=[learning_rate_reduction])

        y_pred = model.predict(X_test)

        print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

        save_model(model, 'resnet_model')

    else:
        #VGG16 parameters
        input_shape = (224,224,3)
        lr = 1e-5
        epochs = 10
        batch_size = 256

        #Building VGG16
        vgg16_model = VGG16(include_top=False, weights= 'imagenet', input_tensor=None, input_shape=input_shape, pooling='avg', classes=2)

        for layer in vgg16_model.layers:
            layer.trainable = False
            
        model = Sequential()
        model.add(vgg16_model)
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer = 'Adam' ,loss = "binary_crossentropy", metrics=["accuracy"])


        run_model(model)

        #Test VGG16
        model.fit(X_train, y_train,
          epochs=epochs, batch_size= epochs, verbose=0,
          callbacks=[learning_rate_reduction]
         )

        y_pred = model.predict(X_test)

        print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

        save_model(model, 'vgg_model')

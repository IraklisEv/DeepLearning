from PIL import Image
import os
import pandas as pd
train_folder = './Training'
valid_folder = './Validation'
small_folder = './small'

train_data = pd.read_csv(train_folder + '/' + 'ISIC2018_Training_GroundTruth.csv')
valid_data = pd.read_csv(valid_folder + '/' + 'ISIC2018_Validation_GroundTruth.csv')

malignant = train_data.loc[(train_data['MEL'] == 1)| (train_data['BCC'] == 1)]['image'].unique()
benign = train_data.loc[(train_data['MEL'] == 0) & (train_data['BCC'] == 0)]['image'].unique()

def separate(dir):
    if not os.path.exists(dir + '/malignant'):
        os.makedirs(dir + '/malignant')

    if not os.path.exists(dir + '/benign'):
        os.makedirs(dir + '/benign')
    
    imglist = os.listdir(dir)
    imglist = [x for x in imglist if '.jpg' in x]    
    
    for i in range(len(imglist)):
        moved = False
        if i % 100 == 0:
            print(f'Moved {i} images out of {len(imglist)}')
        img = Image.open(dir + '/' + imglist[i])
        if imglist[i][:-4] in malignant:
            img.save(dir + '/malignant/' + imglist[i])
            moved = True
        elif imglist[i][:-4] in benign:
            img.save(dir + '/benign/' + imglist[i])
            moved = True
        if moved:
            os.remove(dir + '/' + imglist[i])

if __name__ == '__main__':
    separate(train_folder)
    separate(valid_folder)
import os
from PIL import Image

if __name__ == '__main__':
    imglist = os.listdir('./Training')
    imglist.remove('ISIC2018_Training_GroundTruth.csv')
    print(f'Found {len(imglist)} photos to resize')
    for i in range(len(imglist)):
    	if i % 100 == 0:
    		print(f'Resized {i} images out of {len(imglist)}')
    	img = Image.open('./Validation/' + imglist[i])
    	img = img.resize((112,112))
    	img.save('./res_val/' + imglist[i])


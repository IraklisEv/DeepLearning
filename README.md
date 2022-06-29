# DeepLearning
### Project for the course in Deep Learning, MSc Artificial Intelligence, University of Piraeus/ NCSR Democritos.
## Overview of the project
Implementation and comparison between highly tuned CNN model and the ResNet50 and VGG19 models, on the task of identifying melanoma from dermatoscopic images, provided in the ISIC2018 Dataset.

## Instructions
The following process was applied to the original ISIC2018 training dataset.
1. Run the script separate.py, to separate malignant and benign images into separate folders, create the training and testing subsets.
2. Run the script resize112.py to resize the images to 128x128 resolution.
3. Run the script augment.py to apply data augmentation to the malignant class of the training subset.
4. Open the jupyter notebook and run the code.

## Installation
1. Clone the repo: git clone https://github.com/IraklisEv/DeepLearning.git
2. Install requirements by runing the pip command on requirements.txt
3. Unzip the Dataset.zip file
4. Open the jupyter notebook and run the code.

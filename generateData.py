import os
import numpy as np
import skimage.io as io
import skimage.transform as transform
import time
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage.color import rgba2rgb
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split


def showImage(I):
    plt.imshow(I)
    plt.show()


def addPlainBackground(countryName):

    # get flag
    countryImg = io.imread(f'./flags/{countryName}')/255
    # shrink image
    countryImg = transform.resize(countryImg, (100,100))
    countryImg= rgba2rgb(countryImg)
    countryImg = countryImg+0.09
    l_img = io.imread('blackBackground.jpg')/255
    l_img = transform.resize(l_img, (500,500))
    l_img[200:200+countryImg.shape[0], 200:200+countryImg.shape[1]] = countryImg
    io.imsave(f'flagsPlainBackground/{countryName.replace(".png", ".jpeg")}', l_img)



def foregroundAug(foreground):
    # Random rotation, zoom, translation
    angle = np.random.randint(-10, 10) * (np.pi / 180.0)  # Convert to radians
    zoom = np.random.random() * 0.4 + 0.8  # Zoom in range [0.8,1.2)
    t_x = np.random.randint(0, int(foreground.shape[1] / 3))
    t_y = np.random.randint(0, int(foreground.shape[0] / 3))

    tform = transform.AffineTransform(scale=(zoom, zoom),
                                      rotation=angle,
                                      translation=(t_x, t_y))
    foreground = transform.warp(foreground, tform.inverse)

    # Random horizontal flip with 0.5 probability
    if (np.random.randint(0, 100) >= 50):
        foreground = foreground[:, ::-1]

    return foreground

def getForegroundMask(foreground):
        mask_new = foreground.copy()[:, :, 0]
        mask_new[mask_new > 0] = 1
        return mask_new



# add flag to background
def compose(foreground, mask, background):
    # resize background
    background = transform.resize(background, foreground.shape[:2])

    # Subtract the foreground area from the background
    background = background * (1 - mask.reshape(foreground.shape[0], foreground.shape[1], 1))

    # Finally, add the foreground
    composed_image = background + foreground

    return composed_image

def main():
    imageFiles = [f for f in listdir('./flags') if isfile(join('./flags', f))]
    categories = ['Argentina', 'Austria', 'Bangladesh', 'Canada', 'Croatia', 'Finland', 'Greece', 'Guatemala', 'Honduras', 'India', 'Jamaica', 'Japan', 'Portugal', 'Puerto Rico', 'Rwanda', 'Slovakia', 'Spain', 'UnitedKingdom', 'UnitedStates', 'Wales']

    def addPlainBackgrounds():
        [addPlainBackground(imageFile) for imageFile in imageFiles]
        # Create a mask for this new foreground object

    #addPlainBackgrounds()
    imageFilesBlack = [f for f in listdir('./flagsPlainBackground') if isfile(join('./flagsPlainBackground', f))]

    def generateSamples():
        for imageName in imageFilesBlack:
            for ii in range(100):
                foreground = io.imread('./flagsPlainBackground/'+imageName)/255
                foreground = foregroundAug(foreground)
                foregroundMask = getForegroundMask(foreground/255)
                background_fileName = np.random.choice(os.listdir("./backgrounds/"))
                background = io.imread('./backgrounds/' + background_fileName) / 255.0
                newImage = compose(foreground, foregroundMask, background)
                io.imsave('./samples/'+imageName.replace(".jpeg", str(ii)) +'.jpeg', newImage)

    # reading the samples images and keeping track of the true labels
    samplesImages = [f for f in listdir('./samples/') if isfile(join('./samples/', f))]
    images = []
    labels = []

    for ii, file in enumerate(samplesImages):
        if 'India' in file or 'Spain' in file:
            flattenedImage = io.imread('samples/'+file).flatten()
            images.append(flattenedImage)
            labels.append(categories.index(''.join([i for i in file.replace('.jpeg', "") if not i.isdigit()])))

    flat_images = np.array(images)
    print("Number of images", len(flat_images))
    labels = np.array(labels)



    # creating the model
    param_grid={'C':[0.1,1,10],'gamma':[0.1,1],'kernel':['rbf','poly']}
    svc=svm.SVC(probability=True)
    model=GridSearchCV(svc,param_grid)

    # creating test dataset
    x_train, x_test, y_train, y_test = train_test_split(flat_images, labels, test_size=0.20, random_state=77, stratify=labels)
    print("Test Split")
    model.fit(x_train,y_train)
    print("Model fit")
    y_pred = model.predict(x_test)
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")






if __name__ == "__main__":
    main()
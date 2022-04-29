import os
import numpy as np
import random
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage.color import rgba2rgb
import tensorflow as tf
from keras import layers
from keras.models import Sequential


def showImage(I):
    plt.imshow(I)
    plt.show()


def addPlainBackground(countryName):
    # get flag
    countryImg = io.imread(f'./lowerResolution/{countryName}') / 255
    # shrink image
    countryImg = transform.resize(countryImg, (20, 20))
    countryImg = rgba2rgb(countryImg)
    countryImg = countryImg + 0.09
    l_img = io.imread('blackBackground.jpg') / 255
    l_img = transform.resize(l_img, (100, 100))
    l_img[40:40 + countryImg.shape[0], 40:40 + countryImg.shape[1]] = countryImg
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


def lowerResolutionFile(filename):
    image = io.imread(filename) / 255
    rescaled_image = transform.resize(image, (20, 20))
    return rescaled_image


def main():
    flags = [f for f in listdir('./flags') if isfile(join('./flags', f))]
    imageFiles = [f for f in listdir('./lowerResolution') if isfile(join('./lowerResolution', f))]
    # categories = ['Argentina', 'Austria', 'Bangladesh', 'Canada', 'Croatia', 'Finland', 'Greece', 'Guatemala', 'Honduras', 'India', 'Jamaica', 'Japan', 'Portugal', 'Puerto Rico', 'Rwanda', 'Slovakia', 'Spain', 'UnitedKingdom', 'UnitedStates', 'Wales']
    categories = ['Argentina', 'Austria', 'Bangladesh']

    def lowerResolutions():
        for imageFile in flags:
            lower_res = lowerResolutionFile('./flags/' + imageFile)
            io.imsave('./lowerResolution/' + imageFile, lower_res)

    def addPlainBackgrounds():
        [addPlainBackground(imageFile) for imageFile in imageFiles]
        # Create a mask for this new foreground object


    def generateSamples():
        lowerResolutions()
        addPlainBackgrounds()
        imageFilesBlack = [f for f in listdir('./flagsPlainBackground') if isfile(join('./flagsPlainBackground', f))]
        for imageName in imageFilesBlack:
            if any([country in imageName for country in categories]):
                for ii in range(100):
                    foreground = io.imread('./flagsPlainBackground/' + imageName) / 255
                    foreground = foregroundAug(foreground)
                    foregroundMask = getForegroundMask(foreground / 255)
                    background_fileName = np.random.choice(os.listdir("./backgrounds/"))
                    background = lowerResolutionFile('./backgrounds/' + background_fileName)
                    newImage = compose(foreground, foregroundMask, background)
                    io.imsave('./samples/' + imageName.replace(".jpeg", str(ii)) + '.jpeg', newImage)

    generateSamples()
    # reading the samples images and keeping track of the true labels
    samplesImages = [f for f in listdir('./samples/') if isfile(join('./samples/', f))]
    images = []
    labels = []

    for ii, file in enumerate(samplesImages):
        if any([country in file for country in categories]):
            image = io.imread('samples/' + file)
            images.append(image)
            labels.append(categories.index(''.join([i for i in file.replace('.jpeg', "") if not i.isdigit()])))

    images = np.array(images)
    print("Number of samples", len(images))
    labels = np.array(labels)
    num_classes = len(categories)

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(100, 100, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs = 10
    samples = []
    rand_shuffle = list(range(len(images)))
    random.shuffle(rand_shuffle)
    history = model.fit(images[rand_shuffle[:-20]], labels[rand_shuffle[:-20]],
                        validation_data=(images[rand_shuffle[-20:]], labels[rand_shuffle[-20:]]), epochs=10,
                        batch_size=100)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
    main()

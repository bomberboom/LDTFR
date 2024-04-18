from keras.models import Model
import numpy as np
from scipy import io
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import MaxPooling2D
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import os
import glob
import keras

## modified with your data path
target_size = 224
train_path=r"D:\research\Undergraduate Graduation Project\bishe\paper&data\database\TrainingImages"
test_path=r"D:\research\Undergraduate Graduation Project\bishe\paper&data\database\TestingImages"


train_images = []
test_images = []
train_label = []
test_label = []
file_path = train_path
image_files = glob.glob(os.path.join(file_path, "*.png"))
for image_file in image_files:
    img = image.load_img(image_file, target_size = (target_size, target_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    min_val = np.min(x)
    max_val = np.max(x)
    normalized_arr = (x - min_val) / (max_val - min_val)
    train_images.append(normalized_arr)
    train_label.append(int(image_file[-5]))
train_images = np.concatenate(train_images, axis = 0)
file_path = test_path
image_files = glob.glob(os.path.join(file_path, "*.png"))
for image_file in image_files:
    img = image.load_img(image_file, target_size = (target_size, target_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    min_val = np.min(x)
    max_val = np.max(x)
    normalized_arr = (x - min_val) / (max_val - min_val)
    test_images.append(normalized_arr)
    test_label.append(int(image_file[-5]))
test_images = np.concatenate(test_images, axis = 0)
train_images = np.array(train_images)
train_label = np.array(train_label)
test_label = np.array(test_label)

##
base_model = VGG16(weights = 'imagenet', include_top = True,
                   backend = keras.backend,
                   layers = keras.layers,
                   models = keras.models,
                   utils = keras.utils)

x = base_model.get_layer('block1_pool').output
x = MaxPooling2D(pool_size = (2, 2))(x)
model = Model(inputs = base_model.input, outputs = x)

feature_total_train = []
for i in range(len(train_images)):
    x = train_images[i, :, :, :]
    x = np.expand_dims(x, axis = 0)
    mixed5_features = model.predict(x)
    mixed5_features = np.squeeze(mixed5_features)
    feature_total_train.append(mixed5_features)
    print(i)
feature_total_test = []
for i in range(len(test_images)):
    x = test_images[i, :, :, :]
    x = np.expand_dims(x, axis = 0)
    mixed5_features = model.predict(x)
    mixed5_features = np.squeeze(mixed5_features)
    feature_total_test.append(mixed5_features)
    print(i)
np.save('./T&T_vgg16_train.npy', feature_total_train)
np.save('./T&T_vgg16_test.npy', feature_total_test)
np.save('./T&T_vgg16_train_label.npy', train_label)
np.save('./T&T_vgg16_test_label.npy', test_label)

from datetime import datetime
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0

model_name = 'EfficientNetB0'  # VGG16, NASNet, ResNet50, ResNet101, MobileNet
save_dir = './model_save'

batch_size = 16
FREEZE_LAYERS = 1
epoch_divider = 25
nval = 4000

nrows = 331
ncolumns = 331
channels = 3

today = datetime.today().strftime("%Y%m%d")

from tensorflow.keras.applications import EfficientNetB0

with tf.device('/gpu:0'):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=None,
                                input_shape=(nrows, ncolumns, channels))
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    count = 0
    output_layer = Dense(17, activation='softmax', name='softmax')(x)
    net_final = Model(inputs=base_model.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
        count = count + 1
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    net_final.compile(optimizer=Adam(lr=1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

with tf.device('/gpu:0'):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       width_shift_range=0.2,
                                       # height_shift_range=0.2,
                                       # shear_range=0.2,
                                       # zoom_range=0.2,
                                       horizontal_flip=True,
                                       brightness_range=[0.7, 1.3],
                                       fill_mode='reflect',
                                       validation_split=0.25)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # nrows * ncolumns 크기로 불러오게 됨.
    train_generator = train_datagen.flow_from_directory(
        './barkSNU',
        target_size=(nrows, ncolumns),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    val_generator = train_datagen.flow_from_directory(
        './barkSNU',
        target_size=(nrows, ncolumns),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

tf.debugging.set_log_device_placement(True)
with tf.device('/gpu:0'):
    history = net_final.fit(train_generator,
                            steps_per_epoch = train_generator.n // batch_size,
                            validation_freq=5,
                            epochs = 10,
                            validation_data=val_generator,
                            validation_steps= val_generator.n // batch_size
                            )

    net_final.save(f'./model_save/{model_name}')

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16, InceptionV3, DenseNet201
from tensorflow.keras.utils import to_categorical

def get_base_model(name="vgg16", img_width=224, img_height=224, channels=3):
    input_shape = (img_width, img_height, channels)
    if name == "vgg16":
        return VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif name == "inceptionv3":
        return InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif name == "densenet201":
        return DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Model not supported")

def build_model(base_model, num_classes):
    visible = Input(shape=base_model.input_shape[1:], name='input_layer')
    x = base_model(visible)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='sigmoid', name='output_layer')(x)
    model = Model(inputs=visible, outputs=output)
    return model

def main(args):
    # Load and preprocess data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=(0.73, 0.9),
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.10,
        fill_mode='constant',
        height_shift_range=0.10,
        brightness_range=(0.55, 0.9),
        validation_split=0.2
    )

    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        args.train_data_dir,
        target_size=(args.img_width, args.img_height),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=True,
    )

    valid_generator = valid_test_datagen.flow_from_directory(
        args.val_data_dir,
        target_size=(args.img_width, args.img_height),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=False,
    )

    test_generator = valid_test_datagen.flow_from_directory(
        args.test_data_dir,
        target_size=(args.img_width, args.img_height),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=False,
    )

    # Model creation
    base_model = get_base_model(args.base_model, args.img_width, args.img_height, args.channels)
    base_model.trainable = False  # Freeze base model layers
    model = build_model(base_model, len(train_generator.class_indices))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    ]

    # Training
    model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=len(train_generator.filenames) // args.batch_size,
        validation_data=valid_generator,
        validation_steps=len(valid_generator.filenames) // args.batch_size,
        verbose=1,
        callbacks=callbacks,
        shuffle=True
    )

    # Evaluation
    model.evaluate(test_generator, batch_size=args.batch_size, verbose=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on different datasets.')
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--val_data_dir', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_model', type=str, default='vgg16', choices=['vgg16', 'inceptionv3', 'densenet201'])
    args = parser.parse_args()

    main(args)
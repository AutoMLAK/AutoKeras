import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.models import load_model
import autokeras as ak

def load_images_from_directory(test_data_dir, img_width, img_height):
    x_test = []
    for folder in os.listdir(test_data_dir):
        sub_path = os.path.join(test_data_dir, folder)
        for img_name in os.listdir(sub_path):
            image_path = os.path.join(sub_path, img_name)
            img = image.load_img(image_path, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            x_test.append(img)
    return np.vstack(x_test)

def evaluate_model(model_path, test_data, test_x):
    model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
    y_prob = model.predict(test_x)
    auc = roc_auc_score(test_data.classes, y_prob)
    print('AUC: %.3f' % auc)

    y_pred = np.where(y_prob > 0.5, 1, 0)
    cm = confusion_matrix(test_data.classes, y_pred)
    df_cm = pd.DataFrame(cm, list(test_data.class_indices.keys()), list(test_data.class_indices.keys()))

    fig, ax = plt.subplots(figsize=(10,8))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix
')
    plt.show()

    print('Classification Report
')
    target_names = list(test_data.class_indices.keys())
    print(classification_report(test_data.classes, y_pred, target_names=target_names))

def main(args):
    test_generator = ImageDataGenerator(rescale=1./255)
    test_data = test_generator.flow_from_directory(
        args.test_data_dir,
        target_size=(args.img_width, args.img_height),
        color_mode='rgb',
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=False,
    )
    test_x = load_images_from_directory(args.test_data_dir, args.img_width, args.img_height)
    evaluate_model(args.model_path, test_data, test_x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate AutoKeras model.')
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    main(args)
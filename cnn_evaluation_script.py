import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def load_and_prep_image_data(test_data_dir, img_width, img_height, batch_size):
    test_generator = ImageDataGenerator(rescale=1./255)
    testgen = test_generator.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )
    return testgen

def evaluate_model(model_path, test_data):
    model = load_model(model_path)
    Y_pred = model.predict(test_data)
    auc = roc_auc_score(test_data.classes, Y_pred)
    print('AUC: %.3f' % auc)
    
    y_pred = np.where(Y_pred > 0.5, 1, 0)
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
    test_data = load_and_prep_image_data(args.test_data_dir, args.img_width, args.img_height, args.batch_size)
    evaluate_model(args.model_path, test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CNN model.')
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    main(args)
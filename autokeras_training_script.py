import argparse
import numpy as np
import tensorflow as tf
import autokeras as ak

def create_classifier(metrics, max_trials, project_name):
    return ak.ImageClassifier(overwrite=True, max_trials=max_trials, project_name=project_name, metrics=metrics)

def main(args):
    METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    
    train_data = ak.image_dataset_from_directory(
        args.train_dir,
        image_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
    )

    val_data = ak.image_dataset_from_directory(
        args.val_dir,
        image_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
    )

    test_data = ak.image_dataset_from_directory(
        args.test_dir,
        image_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
    )

    if args.use_multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            clf = create_classifier(METRICS, args.max_trials, args.project_name)
            clf.fit(train_data, epochs=args.epochs, validation_data=val_data)
    else:
        clf = create_classifier(METRICS, args.max_trials, args.project_name)
        clf.fit(train_data, epochs=args.epochs, validation_data=val_data)

    model = clf.export_model()
    print(model.summary())
    model.save(f'{args.model_name}.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on different datasets using AutoKeras.')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_trials', type=int, default=10)
    parser.add_argument('--project_name', type=str, default='autokeras_image_clf')
    parser.add_argument('--model_name', type=str, default='autokeras_model')
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    args = parser.parse_args()

    main(args)
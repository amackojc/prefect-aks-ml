import cv2
import mlflow
import os
import tensorflow as tf
import numpy as np

from prefect import task, flow, tags
from tensorflow.keras.applications.vgg19 import VGG19
from imblearn.combine import SMOTETomek
from imblearn.datasets import make_imbalance
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model


@task(log_prints=False)
def load_data(data_path, image_size):
    data = []
    labels = []
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        for image in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, image))
            img_resized = cv2.resize(img, (image_size, image_size))
            data.append(img_resized)
            labels.append(class_name)

    encoder_labels = LabelEncoder()
    data = np.array(data) / 255  # normalization
    labels = encoder_labels.fit_transform(np.array(labels))
    return data, labels


@task(log_prints=False)
def make_data_imbalanced(X_data, y_data, image_size):
    X_data_imbalanced, y_data_imbalanced = make_imbalance(
        X_data.reshape(X_data.shape[0], 3 * image_size**2),
        y_data, sampling_strategy={0: 350, 1: 3500},
        random_state=42
    )

    return X_data_imbalanced, y_data_imbalanced


def smotetomek_oversampling(X_train, y_train, image_size):
    smt = SMOTETomek(random_state=42)
    X_train_smotetomek, y_train_smotetomek = smt.fit_resample(X_train, y_train)
    X_train_smotetomek = X_train_smotetomek.reshape(
            X_train_smotetomek.shape[0],
            image_size,
            image_size,
            3
    )
    return X_train_smotetomek, y_train_smotetomek


@task(log_prints=True)
def create_vgg19_model(image_size, num_classes=1):
    # Load pre-trained VGG19 model
    base_model = VGG19(weights='imagenet',
                       include_top=False,
                       input_shape=(image_size, image_size, 3))
    # Freeze the base model layers
    base_model.trainable = False
    # Create the sequential model
    model = tf.keras.Sequential([
        # Use the base model as the first layer
        base_model,
        
        # Add custom classification layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])

    model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
    )

    return model


@task(log_prints=True)
def train_model(X_train, y_train, X_val, y_val, model, batch_size, epochs):

    mlflow.set_tracking_uri(uri="http://mlflow.mlflow.svc.cluster.local:5000")
    mlflow.set_experiment("Pneumonia Imbalanced Data - VGG19 SMOTETomek")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )
    return model


@task(log_prints=True)
def evaluate_model(X_val, y_val, X_test, y_test, model):
    with mlflow.start_run():
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_loss", val_loss)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        print(f"Validation Accuracy: {val_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")

        print(f"Validation Loss: {val_loss}")
        print(f"Test Loss: {val_loss}")


@task(log_prints=True)
def predict_model(X_test, y_test, model):
    with mlflow.start_run():
        y_pred = model.predict(X_test)
        y_pred_classes = tf.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        print(f"Prediction -  Accuracy: {accuracy}")
        print(f"Prediction -  Precision: {precision}")
        print(f"Prediction -  Recall: {recall}")
        print(f"Prediction -  F1-score: {f1}")


@flow(log_prints=True)
def imbalanced_vgg19_smotetomek_pipeline():
    image_size = 128
    X_train, y_train = load_data("data/train", image_size)
    X_val, y_val = load_data("data/val", image_size)
    X_test, y_test = load_data("data/test", image_size)
    X_train_imbalanced, y_train_imbalanced = make_data_imbalanced(X_train, y_train, image_size)
    X_train_smotetomek, y_train_smotetomek = smotetomek_oversampling(X_train_imbalanced, y_train_imbalanced, image_size)
    model_design_cnn = create_vgg19_model(image_size)
    model_trained = train_model(X_train_smotetomek, y_train_smotetomek, X_val, y_val, model_design_cnn, 32, 10)
    evaluate_model(X_val, y_val, X_test, y_test, model_trained)
    predict_model(X_test, y_test, model_trained)


if __name__ == "__main__":
    with tags("local"):
        imbalanced_vgg19_smotetomek_pipeline()

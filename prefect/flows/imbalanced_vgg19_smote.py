import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from imblearn.over_sampling import SMOTE
from prefect import task, flow, tags
from imblearn.datasets import make_imbalance
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@task(log_prints=False)
def load_data(dir_path, image_size):
    X_data = []
    y_data = []

    data = image_dataset_from_directory(
        directory=dir_path,
        labels='inferred',
        label_mode='binary',
        batch_size=32,
        image_size=(image_size, image_size)
    )
    for feature, label in data:
        X_data.append(feature.numpy())
        y_data.append(label.numpy())

    X_data = np.concatenate(X_data, axis=0)
    y_data = np.concatenate(y_data, axis=0).flatten()

    return X_data, y_data


@task(log_prints=False)
def make_data_imbalanced(X_data, y_data, samples_ratio, image_size):
    unique_classes, class_counts = np.unique(y_data, return_counts=True)
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_class = unique_classes[np.argmin(class_counts)]

    majority_count = np.max(class_counts)
    minority_count = int(majority_count * samples_ratio)

    sampling_strategy = {
        majority_class: majority_count,
        minority_class: minority_count
    }

    X_data_imbalanced, y_data_imbalanced = make_imbalance(
        X_data.reshape(X_data.shape[0], 3 * image_size**2),
        y_data, sampling_strategy=sampling_strategy,
        random_state=42
    )

    X_data_imbalanced = X_data_imbalanced.reshape(
            X_data_imbalanced.shape[0],
            image_size,
            image_size,
            3
    )

    return X_data_imbalanced, y_data_imbalanced


@task(log_prints=False)
def smote_oversampling(X_train, y_train, image_size):
    smt = SMOTE()
    X_train_smote, y_train_smote = smt.fit_resample(
            X_train.reshape(X_train.shape[0], 3 * image_size**2),
            y_train
    )

    X_train_smote = X_train_smote.reshape(
            X_train_smote.shape[0],
            image_size,
            image_size,
            3
    )

    return X_train_smote, y_train_smote


@task(log_prints=True)
def create_vgg19_model(image_size, num_classes=1):
    # Load pre-trained VGG19 model
    base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])

    model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
    )

    return model


@task(log_prints=True)
def train_model(X_train, y_train, X_val, y_val, model, batch_size, epochs):
    data_gen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        shear_range=0.1
    )

    model.fit(
        data_gen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs
    )

    return model


@task(log_prints=True)
def predict_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0)

    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia']
    )

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_classes, target_names=['NORMAL', 'PHEUMONIA']))


@flow(log_prints=True)
def imbalanced_vgg19_smote_pipeline():
    image_size = 256

    mlflow.set_tracking_uri(uri="http://mlflow.mlflow.svc.cluster.local:5000")
    mlflow.set_experiment("Pneumonia Balance Data - VGG19 Imbalanced SMOTE")
    mlflow.tensorflow.autolog()

    X_data, y_data = load_data("data", image_size)

    print(f'Before imbalancing: {Counter(y_data)}')
    X_data_imbalanced, y_data_imbalanced = make_data_imbalanced(X_data, y_data, 1/9, image_size)

    X_train, X_val, y_train, y_val = train_test_split(
            X_data_imbalanced,
            y_data_imbalanced,
            test_size=0.2,
            random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=123
    )

    X_train = X_train / 255
    X_val = X_val / 255
    X_test = X_test / 255

    print(f'After imbalancing: {Counter(y_train)}')
    X_train_smote, y_train_smote = smote_oversampling(X_train, y_train, image_size)
    print(f'After SMOTE: {Counter(y_train_smote)}')

    with mlflow.start_run():
        model_design_cnn = create_vgg19_model(image_size)
        model_trained = train_model(X_train_smote, y_train_smote, X_val, y_val, model_design_cnn, 32, 5)
        predict_model(X_test, y_test, model_trained)


if __name__ == "__main__":
    with tags("local"):
        imbalanced_vgg19_smote_pipeline()

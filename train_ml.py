import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score

from src.constants import FEATURE_COLUMNS, LABEL, KNOWN_CLASSES, RESULT_DIR


if __name__ == '__main__':
    # Load the data
    train_df = pd.read_csv('dataset/train.csv')
    valid_df = pd.read_csv('dataset/valid.csv')
    test_df = pd.read_csv('dataset/test.csv')

    X_train = train_df[FEATURE_COLUMNS].to_numpy()
    y_train = train_df[LABEL].to_numpy()

    X_valid = valid_df[FEATURE_COLUMNS].to_numpy()
    y_valid = valid_df[LABEL].to_numpy()

    X_test = test_df[FEATURE_COLUMNS].to_numpy()
    y_test = test_df[LABEL].to_numpy()

    # Train a Random Forest model
    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    top1_acc = accuracy_score(y_test, y_pred)
    top3_acc = top_k_accuracy_score(y_test, model.predict_proba(X_test), k=3, labels=KNOWN_CLASSES)
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc * 100:.2f}%")

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=KNOWN_CLASSES, yticklabels=KNOWN_CLASSES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix_plot.png'))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate_model(y_true, y_pred, class_names):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print("\nClassification Report")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    return cm
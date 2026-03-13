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

    cm = confusion_matrix(y_true, y_pred)

    # 返回混淆矩阵与文本报告，不在此打印
    return cm, report
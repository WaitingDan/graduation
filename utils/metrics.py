from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np


def evaluate_model(y_true, y_pred, class_names):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    cm = confusion_matrix(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # per-class support (真实样本数)
    support = cm.sum(axis=1)

    # per-class recall (正确预测为该类的数量 / 该类真实样本数)
    # 这通常被视为对每一类的识别准确率（class-wise recall）
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_recall = np.diag(cm) / support
        per_class_recall = np.where(np.isnan(per_class_recall), 0.0, per_class_recall)

    # per-class overall accuracy: (TP + TN) / total_samples
    total = cm.sum()
    tp = np.diag(cm)
    # TN = total - (sum of row i + sum of col i - TP)
    tn = total - (cm.sum(axis=1) + cm.sum(axis=0) - tp)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_overall_acc = (tp + tn) / total if total > 0 else np.zeros_like(tp, dtype=float)

    # 返回混淆矩阵、文本报告、每类召回（作为每类识别准确率）、每类整体准确率与抗不均衡指标
    return cm, report, per_class_recall, per_class_overall_acc, support, macro_f1, balanced_acc
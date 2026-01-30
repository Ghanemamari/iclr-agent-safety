# from typing import Dict
# import numpy as np
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support

# def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
#     preds = (probs >= threshold).astype(int)
#     auroc = roc_auc_score(y_true, probs) if len(set(y_true.tolist())) > 1 else float("nan")
#     acc = accuracy_score(y_true, preds)
#     f1 = f1_score(y_true, preds, zero_division=0)
#     p, r, f1b, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
#     return {
#         "auroc": float(auroc),
#         "accuracy": float(acc),
#         "f1": float(f1),
#         "precision": float(p),
#         "recall": float(r),
#         "threshold": float(threshold),
#     }


from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support

def compute_metrics(y_true, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)

    y_list = y_true.tolist() if hasattr(y_true, "tolist") else list(y_true)
    auroc = roc_auc_score(y_true, probs) if len(set(y_list)) > 1 else float("nan")

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    p, r, f1b, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)

    return {
        "auroc": float(auroc),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(threshold),
    }

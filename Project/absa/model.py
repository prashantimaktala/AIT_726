from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    return pd.DataFrame([
        {'metric': 'accuracy', 'score': accuracy},
        {'metric': 'precision_macro', 'score': precision_macro},
        {'metric': 'precision_micro', 'score': precision_micro},
        {'metric': 'recall_macro', 'score': recall_macro},
        {'metric': 'recall_micro', 'score': recall_micro},
        {'metric': 'f1_macro', 'score': f1_macro},
        {'metric': 'f1_micro', 'score': f1_micro},
    ])

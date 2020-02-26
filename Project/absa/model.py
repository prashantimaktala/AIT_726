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
        {'index': 'accuracy', 'value': accuracy},
        {'index': 'precision_macro', 'value': precision_macro},
        {'index': 'precision_micro', 'value': precision_micro},
        {'index': 'recall_macro', 'value': recall_macro},
        {'index': 'recall_micro', 'value': recall_micro},
        {'index': 'f1_macro', 'value': f1_macro},
        {'index': 'f1_micro', 'value': f1_micro},
    ]).set_index('index')['value']

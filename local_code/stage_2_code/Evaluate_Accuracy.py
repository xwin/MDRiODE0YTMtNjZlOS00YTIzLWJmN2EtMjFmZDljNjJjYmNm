'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        #print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])


class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        return precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0)


class Evaluate_Recall(evaluate):
    data = None

    def evaluate(self):
        return recall_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0)


class Evaluate_F1(evaluate):
    data = None

    def evaluate(self):
        return f1_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0)


class Evaluate_All(evaluate):
    data = None

    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print("\n--- Evaluation Summary ---")
        print(f"Accuracy:           {acc:.4f}")
        print(f"Weighted Precision: {prec:.4f}")
        print(f"Weighted Recall:    {rec:.4f}")
        print(f"Weighted F1:        {f1:.4f}")

        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
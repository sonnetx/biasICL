import pandas as pd 
from sklearn.metrics import f1_score
import numpy as np

data = pd.read_csv("C:/Users/sonne/Downloads/chexpert_0_0_gpt-4o-2024-05-13_50.csv")

ground_truth = data['ground_truth']
predictions = data['parsed_answer']

# Calculate the accuracy    
accuracy = sum(ground_truth == predictions) / len(ground_truth)

print(f"Accuracy: {accuracy:.2f}")

row = data.iloc[0]['ground_truth']
print(eval(row.replace('.', ',')))
ground_truth = pd.concat([pd.Series(eval(row.replace('.', ',')), dtype=object) for row in ground_truth])
print(ground_truth)
predicted = pd.concat([pd.Series(eval(row.replace('.', ',')), dtype=object) for row in predictions])

print(predicted)

accuracy = sum(gt == pred for gt, pred in zip(ground_truth, predicted)) / len(ground_truth)

ground_truth = np.array([eval(row.replace('.', ',')) for row in data['ground_truth']])
print(ground_truth.shape, ground_truth)
predicted = np.array([eval(row.replace('.', ',')) for row in data['parsed_answer']])
print(predicted.shape, predicted)

# Calculate the macro F1 score (per class)
macro_f1 = f1_score(ground_truth, predicted, average='macro')


print(f"Macro F1: {macro_f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
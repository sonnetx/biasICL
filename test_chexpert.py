import pandas as pd 

data = pd.read_csv("C:/Users/sonne/Downloads/chexpert_0_0_gpt-4o-2024-05-13_50.csv")

ground_truth = data['ground_truth']
predictions = data['parsed_answer']

# Calculate the accuracy    
accuracy = sum(ground_truth == predictions) / len(ground_truth)

print(f"Accuracy: {accuracy:.2f}")

ground_truth = pd.concat([pd.Series(eval(row.replace('.', ',')), dtype=object) for row in ground_truth])
predicted = pd.concat([pd.Series(eval(row.replace('.', ',')), dtype=object) for row in predictions])

accuracy = sum(gt == pred for gt, pred in zip(ground_truth, predicted)) / len(ground_truth)

print(f"Accuracy: {accuracy:.2f}")
import csv
import pandas as pd
ai_data = pd.read_csv('../csv/ai_metadata.csv')
human_data = pd.read_csv('../csv/human_metadata.csv')

with open('../csv/dataset.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["problem_id", "language", "code_size", "label", "model"])
    for row in ai_data.itertuples():
        problem_id = row[1]
        language = row[2]
        code_size = row[3]
        label = row[4]
        model = row[5]
        writer.writerow([problem_id, language, code_size, label, model])

    for row in human_data.itertuples():
        problem_id = row[2]
        language = row[3]
        code_size = row[4]
        label = row[5]
        model = row[6]
        writer.writerow([problem_id, language, code_size, label, model])

import glob
import os
import csv

# 모든 p**** 디렉토리 경로 수집
directories = glob.glob("./ai_codes/p*/")

# 결과를 저장할 리스트
all_file_paths = []

# 각 디렉토리에서 파일 경로 수집
for d in directories:
    files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    for f in files:
        all_file_paths.append(os.path.join(d, f))  # 전체 경로 저장

# CSV 작성
with open('../csv/ai_metadata.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["problem_id", "language", "code_size", "label", "model"])
    for path in all_file_paths:
        filename = os.path.basename(path)
        try:
            problem_id = filename.split("_")[0]
            model = filename.split("_")[1].split(".")[0]
            extension = filename.split(".")[1]
            language = "C++" if extension == "cpp" else "Python"
            code_size = os.stat(path).st_size
            writer.writerow([problem_id, language, code_size, 1, model])
        except Exception as e:
            print(f"Error processing file {path}: {e}")
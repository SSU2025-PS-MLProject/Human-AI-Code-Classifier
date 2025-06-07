from typing import Union
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
import logging
import os
from joblib import load
from python_utils import extract_python_features_from_file
from cpp_utils import extract_cpp_features_from_file
from pathlib import Path
from enum import Enum
import uuid


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() # 콘솔에다가 출력
    ]
)

PYTHON_FEATURES = [
    "avg_identifier_length",
    "average_function_length",
    "token_count",
    "function_count",
    "blank_ratio",
    "identifier_count",
    "total_lines",
    "code_size",
    "max_control_depth",
    "comment_ratio",
] + [f'codebert_{i}' for i in range(768)]

CPP_FEATURES = [
    "code_size",
    "total_lines",
    "blank_ratio",
    "comment_ratio",
    "num_funcs",
    "avg_func_length",
    "max_control_depth",
    "control_count",
    "unique_identifiers",
    "token_count"
] + [f'vec_{i}' for i in range(768)]

logger = logging.getLogger("[FASTAPI]")
logger.info("FASTAPI 어플리케이션 실행..")

logger.info("모델 로드 중...")
python_multilabel_classifier = load("../models/python_xgb_top2.joblib")
python_binary_classifier = load("../models/svm_python_bin.joblib")
cpp_multilabel_classifier = load("../models/cpp_xgb_top2.joblib")
cpp_binary_classifier = load("../models/svm_binary_cpp.joblib")
logger.info("모델 로드 완료")

class Language(str, Enum):
    PYTHON = "python"
    CPP = "cpp"

@app.get("/")
async def root():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.post("/code")
def upload_code(language: Language = Form(), code_file: UploadFile = File()):
    try:
        request_id: str = uuid.uuid4()
        if language == Language.PYTHON:
            content = code_file.file.read()
            if not code_file.filename.endswith('.py'):
                logger.error("Python 파일 확장자가 잘못되었습니다.")
                return JSONResponse(
                    content={
                        "message": "Python 파일은 .py 확장자를 가져야 합니다."
                    },
                    status_code=400
                )
            with open(f"code_{request_id}.py", "wb") as f:
                f.write(content)
        else:
            content = code_file.file.read()
            if not code_file.filename.endswith('.cpp'):
                logger.error("C++ 파일 확장자가 잘못되었습니다.")
                return JSONResponse(
                    content={
                        "message": "C++ 파일은 .cpp 확장자를 가져야 합니다."
                    },
                    status_code=400
                )
            with open(f"code_{request_id}.cpp", "wb") as f:
                f.write(content)

        return JSONResponse(
            content={
                "message": "코드 파일이 성공적으로 업로드되었습니다.",
                "file_id": str(request_id)
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"코드 파일 업로드 중 오류 발생: {e}")
        return JSONResponse(
            content={
                "message": "코드 파일 업로드 중 오류 발생",
                "error": str(e)
            },
            status_code=500
        )

@app.get("/classification")
def classification(file_id: str, language: Language):
    try:
        logger.info("분류 처리 시작")

        if language == Language.PYTHON:
            file_path_str = os.path.join(os.getcwd(), f"code_{file_id}.py")
            file_path_obj = Path(file_path_str)
            logger.info("Python 코드 파일에서 특징 추출중")
            features = extract_python_features_from_file(file_path_obj, logger)
            if features.empty:
                logger.error("Python 코드 파일에서 특징을 추출할 수 없습니다.")
                return JSONResponse(
                    content={
                        "message": "코드 파일에서 특징을 추출할 수 없습니다."
                    },
                    status_code=400
                )
            logger.info(features)
            X_pred = features.reindex(columns=PYTHON_FEATURES)
            logger.info("Python 특징 추출 완료")

            # 다중 레이블 분류 예측
            logger.info("다중 레이블 분류 예측 시작")
            multi_label_probs = python_multilabel_classifier.predict_proba(X_pred)[0]
            multi_label_prediction = multi_label_probs.tolist()
            logger.info("다중 레이블 분류 예측 완료")

            # 이진 분류 예측
            logger.info("이진 분류 예측 시작")
            binary_probs = python_binary_classifier.predict_proba(X_pred)[0]
            binary_prediction = binary_probs.tolist()
            logger.info("이진 분류 예측 완료")
        elif language == Language.CPP:
            file_path_str = os.path.join(os.getcwd(), f"code_{file_id}.cpp")
            file_path_obj = Path(file_path_str)
            logger.info("CPP 코드 파일에서 특징 추출중")
            features = extract_cpp_features_from_file(file_path_obj)
            if features.empty:
                logger.error("CPP 코드 파일에서 특징을 추출할 수 없습니다.")
                return JSONResponse(
                    content={
                        "message": "코드 파일에서 특징을 추출할 수 없습니다."
                    },
                    status_code=400
                )
            logger.info(features)
            X_pred = features.reindex(columns=CPP_FEATURES)
            logger.info("CPP 특징 추출 완료")

            # 다중 레이블 분류 예측
            logger.info("다중 레이블 분류 예측 시작")
            multi_label_probs = cpp_multilabel_classifier.predict_proba(X_pred)[0]
            multi_label_prediction = multi_label_probs.tolist()
            logger.info("다중 레이블 분류 예측 완료")

            # 이진 분류 예측
            logger.info("이진 분류 예측 시작")
            binary_probs = cpp_binary_classifier.predict_proba(X_pred)[0]
            binary_prediction = binary_probs.tolist()
            logger.info("이진 분류 예측 완료")
        else:
            logger.error("지원하지 않는 언어입니다.")
            return JSONResponse(content={"message": "지원하지 않는 언어입니다."}, status_code=400)
        
        multi_label_mapper: dict = {
            "0": "HUMAN",
            "1": "DEEPSEEK",
            "2": "GEMINI",
            "3": "GPT",
            "4": "GROK",
            "5": "MISTRAL"
        }
        for i in range(len(multi_label_prediction)):
            multi_label_prediction[i] = {
                "label": multi_label_mapper[str(i)],
                "probability": multi_label_prediction[i]
            }

        binary_prediction_mapper: dict = {
            "0": "HUMAN",
            "1": "AI"
        }
        for i in range(len(binary_prediction)):
            binary_prediction[i] = {
                "label": binary_prediction_mapper[str(i)],
                "probability": binary_prediction[i]
            }
        
        logger.info("분류 처리 완료")
        return JSONResponse(
            content={
                "multi_label_prediction": multi_label_prediction,
                "binary_prediction": binary_prediction
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"분류 처리 중 오류 발생: {e}")
        return JSONResponse(
            content={
                "message": "분류 처리 중 오류 발생",
                "error": str(e)
            },
            status_code=500
        )
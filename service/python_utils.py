import ast, keyword, builtins, tokenize, csv, os
from pathlib import Path
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

from joblib import load #저장된 sklearn 파이프라인 로드용


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
CODEBERT = RobertaModel.from_pretrained("microsoft/codebert-base").to(DEVICE).eval()

class LoopVisitor(ast.NodeVisitor):
    def __init__(self):
        self.max_depth = 0
        self.cur_depth = 0
    def _visit_loop(self, node):
        self.cur_depth += 1
        self.max_depth = max(self.max_depth, self.cur_depth)
        self.generic_visit(node)
        self.cur_depth -= 1
    visit_For = visit_While = visit_AsyncFor = _visit_loop


class FeatureExtractor:

    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    # Scalar features
    SCALAR_FEATURES = [
        "avg_identifier_length",
        "average_function_length",
        "token_count",
        "function_count",
        "blank_ratio",
        "identifier_count",
        "total_lines",
        "comment_ratio",
        "max_control_depth",
    ]
    # CODEBERT columns
    CODEBERT_COLS = [f"codebert_{i}" for i in range(768)]
    ALL_COLS = SCALAR_FEATURES + CODEBERT_COLS

    def __call__(self, code: str) -> Dict[str, float]:
        """
        code 한 덩어리를 넣으면 {feature_name: value, …} dictionary 리턴
        """
        ft = dict()
        # AST 기반 특징
        ft["avg_identifier_length"]  = self.avg_identifier_length(code)
        ft["average_function_length"] = self.average_function_length(code)
        ft["token_count"]            = self.token_count(code)
        ft["function_count"]         = self.function_count(code)
        ft["blank_ratio"]            = self.blank_ratio(code)
        ft["identifier_count"]       = self.identifier_count(code)
        ft["total_lines"]            = self.total_lines(code)
        ft["comment_ratio"]          = self.comment_ratio(code)
        ft["max_control_depth"]      = self.max_control_depth(code)

        # CodeBERT로 생성한 임베딩 벡터 특징
        emb = self.codebert_embedding(code)          # (768,)
        ft.update({c: v for c, v in zip(self.CODEBERT_COLS, emb)})

        return ft

    # 개별 정적 특징 함수 추출 함수
    def avg_identifier_length(self, s: str) -> float:
        try:
            tree = ast.parse(s); names=set()
            class V(ast.NodeVisitor):
                visit_Name = lambda self,n:(names.add(n.id), self.generic_visit(n))
                def visit_FunctionDef(self,n): names.add(n.name); self.generic_visit(n)
                def visit_ClassDef(self,n): names.add(n.name); self.generic_visit(n)
            V().visit(tree)
            return np.mean([len(n) for n in names]) if names else 0.0
        except SyntaxError: return 0.0

    def average_function_length(self, s: str) -> float:
        try:
            tree=ast.parse(s); lens=[]
            for n in ast.walk(tree):
                if isinstance(n, ast.FunctionDef):
                    end=n.lineno
                    for sub in ast.walk(n):
                        if hasattr(sub,'lineno'): end=max(end,sub.lineno)
                    lens.append(end-n.lineno+1)
            return np.mean(lens) if lens else 0.0
        except SyntaxError: return 0.0

    def token_count(self,s:str)->int:
        try:
            skip={tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,
                  tokenize.INDENT,tokenize.DEDENT,tokenize.ENCODING,tokenize.ENDMARKER}
            return sum(1 for t in tokenize.generate_tokens(StringIO(s).readline)
                       if t.type not in skip and not (t.type==tokenize.ERRORTOKEN and t.string.isspace()))
        except (tokenize.TokenError,IndentationError): return 0

    def function_count(self,s:str)->int:
        try: return sum(isinstance(n,ast.FunctionDef) for n in ast.walk(ast.parse(s)))
        except SyntaxError: return 0

    def blank_ratio(self,s:str)->float:
        lines=s.splitlines(); return sum(not l.strip() for l in lines)/len(lines) if lines else 0.0

    def identifier_count(self,s:str)->int:
        try:
            tree=ast.parse(s); ids=set(); kw=set(keyword.kwlist); bi=set(dir(builtins))
            class V(ast.NodeVisitor):
                def visit_Name(self,n):
                    if n.id not in kw|bi: ids.add(n.id); self.generic_visit(n)
                def visit_FunctionDef(self,n):
                    if n.name not in kw|bi: ids.add(n.name)
                    for a in n.args.args+[n.args.vararg,n.args.kwarg,*n.args.kwonlyargs]:
                        if a and a.arg not in kw|bi: ids.add(a.arg)
                    self.generic_visit(n)
                visit_AsyncFunctionDef=visit_FunctionDef
                def visit_ClassDef(self,n):
                    if n.name not in kw|bi: ids.add(n.name); self.generic_visit(n)
                def visit_Attribute(self,n):
                    if n.attr not in kw|bi: ids.add(n.attr); self.visit(n.value)
            V().visit(tree); return len(ids)
        except SyntaxError: return 0

    def total_lines(self,s:str)->int: return len(s.splitlines())

    def comment_ratio(self,s:str)->float:
        try:
            toks=list(tokenize.generate_tokens(StringIO(s).readline))
            comments=sum(t.type==tokenize.COMMENT for t in toks)
            code=sum(t.type not in {tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,
                                    tokenize.DEDENT,tokenize.ENCODING,tokenize.ENDMARKER}
                     for t in toks)
            return comments/code if code else 0.0
        except (tokenize.TokenError,IndentationError): return 0.0

    def max_control_depth(self,s:str)->int:
        try: v=LoopVisitor(); v.visit(ast.parse(s)); return v.max_depth
        except SyntaxError: return 0

    # -------- CodeBERT 임베딩 --------
    def codebert_embedding(self,code:str)->np.ndarray:
        self.logger.info("CodeBERT 임베딩 추출 시작")
        with torch.no_grad():
            inputs={k:v.to(DEVICE) for k,v in TOKENIZER(code,return_tensors='pt',
                                                         max_length=512,truncation=True).items()}
            out = CODEBERT(**inputs).last_hidden_state[0,0]
        self.logger.info("CodeBERT 임베딩 추출 완료")
        return out.cpu().numpy()

def extract_python_features_from_file(path: Path, logger: logging.Logger) -> pd.DataFrame:
    code = path.read_text(encoding="utf-8", errors="ignore")

    logger.info("FeatureExtractor 인스턴스 생성")
    fe   = FeatureExtractor(logger)

    logger.info("코드 특징 추출")
    row  = fe(code)

    row['code_size'] = path.stat().st_size
    print(row['code_size'])

    # 컬럼 순서를 맞추기 위해 DataFrame으로 반환
    logger.info("DataFrame으로 변환 및 반환")
    return pd.DataFrame([row])

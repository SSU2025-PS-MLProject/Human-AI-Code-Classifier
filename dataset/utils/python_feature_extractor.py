import ast
import keyword
import builtins
import tokenize
from io import StringIO
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()  # Disable dropout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class LoopVisitor(ast.NodeVisitor):
    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0

    def _visit_loop(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)  # Visit children within the loop
        self.current_depth -= 1

    def visit_For(self, node):
        self._visit_loop(node)

    def visit_While(self, node):
        self._visit_loop(node)

    def visit_AsyncFor(self, node):
        self._visit_loop(node)


class FeatureExtractor:
    def __init__(self):
        self.features = [
            "avg_identifier_length",
            "function_length",
            "token_count",
            "function_count",
            "blank_ratio",
            "identifier_count",
            "total_lines",
            "codebert_embedding"
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.model.eval()

    def get_avg_identifier_length(self, target: str) -> float:
        """
        평균 식별자 길이
        """
        try:
            tree = ast.parse(target)
            identifiers = set()

            class IdentifierLengthVisitor(ast.NodeVisitor):
                def visit_Name(self, node):
                    identifiers.add(node.id)
                    self.generic_visit(node)

                def visit_FunctionDef(self, node):
                    identifiers.add(node.name)
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    identifiers.add(node.name)
                    self.generic_visit(node)

            IdentifierLengthVisitor().visit(tree)

            if not identifiers:
                return 0.0

            total_length = sum(len(name) for name in identifiers)
            return total_length / len(identifiers)
        except SyntaxError:
            return 0.0

    def get_function_length(self, target: str) -> float:
        """
        평균 함수 길이
        함수가 없으면 0.0 반환
        """
        try:
            tree = ast.parse(target)
            lengths = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Calculate end line more robustly
                    end_lineno = node.lineno
                    for sub_node in ast.walk(node):
                        if hasattr(sub_node, 'lineno'):
                            end_lineno = max(end_lineno, sub_node.lineno)
                    lengths.append(end_lineno - node.lineno + 1)
            return sum(lengths) / len(lengths) if lengths else 0.0
        except SyntaxError:
            return 0.0

    def get_token_count(self, target: str) -> int:
        """
        코드 전체 토큰 수 반환하는 함수
        주석, 공백, 줄바꿈 토큰 제외
        """
        try:
            tokens = tokenize.generate_tokens(StringIO(target).readline)
            # Define tokens to skip more comprehensively
            skip_tokens = {
                tokenize.COMMENT,
                tokenize.NL,  # Non-logical newline (e.g. inside parentheses)
                tokenize.NEWLINE,  # Logical newline
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,  # Usually at the start, e.g. '# -*- coding: utf-8 -*-'
                tokenize.ENDMARKER  # Marks the end of the file
            }
            # Additionally, filter out whitespace tokens if any are generated (usually not explicitly)
            # tokenize.generate_tokens already skips most physical whitespace between tokens

            count = 0
            for tok in tokens:
                if tok.type not in skip_tokens:
                    # Filter out tokens that are purely whitespace, though `generate_tokens` usually handles this.
                    # For example, a `tokenize.SPACE` type does not exist; spaces separate other tokens.
                    # An ` tokenize.ERRORTOKEN` might represent things like standalone backslashes or invalid indent.
                    if tok.type == tokenize.ERRORTOKEN and tok.string.isspace():
                        continue
                    count += 1
            return count
        except (tokenize.TokenError, IndentationError):
            return 0

    def get_function_count(self, target: str) -> int:
        """
        함수 정의 개수
        """
        try:
            tree = ast.parse(target)
            return sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        except SyntaxError:
            return 0

    def get_blank_ratio(self, target: str) -> float:
        lines = target.splitlines()
        return sum(not line.strip() for line in lines) / len(lines) if lines else 0.0

    def get_identifier_count(self, target: str) -> int:
        """
        고유 식별자 개수 반환하는 함수
        - 파이썬 키워드, 내장 함수는 제외
        - 변수, 함수명, 매개변수, 속성 이름 등 포함
        """
        try:
            tree = ast.parse(target)
            names = set()
            kw = set(keyword.kwlist)
            built_in = set(dir(builtins))  # Using set(dir(builtins)) is fine for common builtins

            class IdentifierVisitor(ast.NodeVisitor):
                def visit_Name(self, node):
                    """
                    식별자 수집
                    """
                    if node.id not in kw and node.id not in built_in:
                        names.add(node.id)
                    self.generic_visit(node)

                def visit_FunctionDef(self, node):
                    # 함수 정의 식별자 수집
                    if node.name not in kw and node.name not in built_in:
                        names.add(node.name)
                    for arg_node in node.args.args:
                        if arg_node.arg not in kw and arg_node.arg not in built_in:
                            names.add(arg_node.arg)
                    if node.args.vararg and node.args.vararg.arg not in kw and node.args.vararg.arg not in built_in:
                        names.add(node.args.vararg.arg)
                    if node.args.kwarg and node.args.kwarg.arg not in kw and node.args.kwarg.arg not in built_in:
                        names.add(node.args.kwarg.arg)
                    for arg_node in node.args.kwonlyargs:
                        if arg_node.arg not in kw and arg_node.arg not in built_in:
                            names.add(arg_node.arg)
                    self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node):
                    # Async 함수 정의 식별자 수집
                    self.visit_FunctionDef(node)

                def visit_ClassDef(self, node):
                    # 클래스 정의 식별자 수집
                    if node.name not in kw and node.name not in built_in:
                        names.add(node.name)
                    self.generic_visit(node)

                def visit_arg(self, node):
                    # 매개변수 식별자 수집
                    if node.arg not in kw and node.arg not in built_in:
                        names.add(node.arg)
                    self.generic_visit(node)

                def visit_Attribute(self, node):
                    # 속성 식별자 수집
                    if node.attr not in kw and node.attr not in built_in:
                        names.add(node.attr)
                    self.visit(node.value)

            IdentifierVisitor().visit(tree)
            return len(names)
        except SyntaxError:
            return 0

    def get_total_lines(self, target: str) -> int:
        return len(target.splitlines())

    def get_codebert_embedding(self, code: str) -> list:
        inputs = self.tokenizer(code, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        codebert_embedding = outputs.last_hidden_state[0, 0]
        return codebert_embedding.cpu().numpy().tolist()
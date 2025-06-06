from __future__ import annotations
import csv
import glob
from pathlib import Path
from typing import Iterable, List
import pandas as pd

BASE_DIR = Path("dataset")
AI_CODES_DIR = BASE_DIR / "ai_codes"
HUMAN_CODES_DIR = BASE_DIR / "python_human_codes"
HUMAN_META_CSV = BASE_DIR / "csv" / "human_metadata.csv"
OUTPUT_CSV = BASE_DIR / "csv" / "final" / "python_dataset.csv"

EXTRACTOR = FeatureExtractor()
HEADER: List[str] = (
    ["problem_id", "language", "code_size", "label", "model"]
    + EXTRACTOR.features
)

ENCODINGS = ("utf-8", "latin1", "cp949")

def read_code(path: Path) -> str:
    """
    주어진 경로에서 코드를 읽어오는 함수 (여러 인코딩으로 읽기 시도)
    """
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Cannot decode {path}")


def iter_ai_files() -> Iterable[Path]:
    """
    p****/ 디렉토리에서 모든 Python 파일을 읽어서 경로를 생성하는 제너레이터 함수
    :return:
    """
    pattern = str(AI_CODES_DIR / "p*" / "*.py")
    yield from (Path(p) for p in glob.glob(pattern))


def parse_ai_filename(path: Path) -> tuple[str, str]:
    name_parts = path.stem.split("_", 1)
    problem_id = name_parts[0]
    model = name_parts[1] if len(name_parts) == 2 else "unknown"
    return problem_id, model


def row_from_code(
    *, problem_id: str | None, language: str, code_size: int, label: int,
    model: str, code: str
) -> List:
    """Assemble CSV row with extracted features."""
    features = [getattr(EXTRACTOR, f"get_{f}")(code) for f in EXTRACTOR.features]
    return [problem_id, language, code_size, label, model] + features


def generate_ai_rows() -> Iterable[List]:
    """
    ai_codes 디렉토리에서 모든 Python 파일을 읽어서 CSV 행을 생성하는 함수
    :return:
    """
    for path in iter_ai_files():
        problem_id, model = parse_ai_filename(path)
        code = read_code(path)
        yield row_from_code(
            problem_id=problem_id,
            language="Python",
            code_size=path.stat().st_size,
            label=1,
            model=model,
            code=code,
        )


def generate_human_rows() -> Iterable[List]:
    """
    python_human_codes 디렉토리에서 모든 Python 파일을 읽어서 CSV 행을 생성하는 함수
    :return:
    """
    meta_df = pd.read_csv(HUMAN_META_CSV, dtype=str)

    for _, row in meta_df.iterrows():
        if row.get("language") == "Python":
            code_path = HUMAN_CODES_DIR / f"{row['submission_id']}.py"
            if not code_path.exists():
                print(f"[WARN] 코드 파일 없음 → {code_path}")
                continue

            code = read_code(code_path)
            yield row_from_code(
                problem_id=row.get("problem_id"),
                language=row.get("language", "Python"),
                code_size=code_path.stat().st_size,
                label=int(row.get("label", 0)),
                model=row.get("Model", "human"),
                code=code,
            )

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for row in generate_ai_rows():
            writer.writerow(row)

        for row in generate_human_rows():
            writer.writerow(row)

    print(f"[INFO] Dataset written → {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
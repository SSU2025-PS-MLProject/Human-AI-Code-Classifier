import csv
import torch
from transformers import RobertaTokenizer, RobertaModel
from pathlib import Path
from clang import cindex
import sys
import os
import pandas as pd


# --- CodeBERT Model and Tokenizer Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = None
MODEL = None

def load_codebert_model():
    global TOKENIZER, MODEL
    try:
        print(f"[INFO] Using device: {DEVICE}")
        print("[INFO] Loading CodeBERT tokenizer and model (microsoft/codebert-base)...")
        TOKENIZER = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        MODEL = RobertaModel.from_pretrained("microsoft/codebert-base")
        MODEL.to(DEVICE)
        MODEL.eval()
        print("[INFO] CodeBERT model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load CodeBERT model or tokenizer: {e}")
        print("Please ensure you have an internet connection, the 'transformers' and 'torch' libraries are installed, "
              "and the model name 'microsoft/codebert-base' is correct.")
        sys.exit(1)

# --- Syntactic Feature Extraction ---
def extract_syntactic_features(file_path):
    # Initialize features with default values
    features = {
        'total_lines': 0,
        'blank_ratio': 0.0,
        'comment_ratio': 0.0,
        'num_funcs': 0,
        'avg_func_length': 0.0,
        'max_control_depth': 0,
        'control_count': 0,
        'unique_identifiers': 0,
        'token_count': 0
    }
    try:
        # 1) Read file lines for line-based metrics
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        features['total_lines'] = total_lines
        if total_lines == 0:
            print("[INFO] File is empty. Syntactic features will be zero.")
            return features

        blank_lines = sum(1 for l in lines if l.strip() == '')
        features['blank_ratio'] = round(blank_lines / total_lines, 4)

        # 2) Calculate comment ratio (simple line-based counting)
        comment_lines = 0
        in_block_comment = False
        for line_content in lines:
            stripped_line = line_content.strip()
            if in_block_comment:
                comment_lines += 1
                if '*/' in stripped_line:
                    in_block_comment = False
                continue
            if stripped_line.startswith('//'):
                comment_lines += 1
            elif '/*' in stripped_line:
                comment_lines += 1
                if '*/' not in stripped_line: # Check if block comment ends on the same line
                    in_block_comment = True
        features['comment_ratio'] = round(comment_lines / total_lines, 4)

        # 3) Clang AST Parsing for more complex features
        index = cindex.Index.create()
        # For a single file, typically only standard C++ version is needed.
        # For files with specific includes, args might need adjustment (e.g., adding -I<include_path>)
        parse_args = ['-std=c++17'] 
        
        tu = None
        try:
            tu = index.parse(file_path, args=parse_args)
        except cindex.LibclangError as e:
            print(f"[WARNING] Clang parsing error for {file_path}: {e}. AST-based features might be zero or inaccurate.")
            return features # Return line-based features if parsing fails severely

        if not tu:
            print(f"[WARNING] Clang Translation Unit is None for {file_path}. AST-based features will be limited.")
            return features # Return line-based features

        # Check for significant parsing errors
        has_errors = any(diag.severity >= cindex.Diagnostic.Error for diag in tu.diagnostics)
        if has_errors:
            print(f"[WARNING] Clang reported parsing errors for {file_path}. AST-based features may be incomplete or inaccurate.")
            # for diag in tu.diagnostics:
            #     if diag.severity >= cindex.Diagnostic.Error:
            #         print(f"  [Clang Error] {diag.spelling} at {diag.location}")

        # AST traversal variables
        _control_count = 0
        _max_depth = 0
        _defined_funcs = [] # Store FUNCTION_DECL nodes that are definitions

        def traverse_ast(node, current_nesting_level=0):
            nonlocal _control_count, _max_depth, _defined_funcs
            
            kind = node.kind

            # Count function definitions
            if kind == cindex.CursorKind.FUNCTION_DECL and node.is_definition():
                _defined_funcs.append(node)
            
            # Check for control flow statements
            is_control_structure = kind in (
                cindex.CursorKind.IF_STMT,
                cindex.CursorKind.FOR_STMT,
                cindex.CursorKind.WHILE_STMT,
                cindex.CursorKind.SWITCH_STMT,
                cindex.CursorKind.DO_STMT
            )

            if is_control_structure:
                _control_count += 1
                # current_nesting_level is the depth *at which* this control structure appears.
                # Its own block introduces a new level, so depth is current_nesting_level + 1.
                _max_depth = max(_max_depth, current_nesting_level + 1)
                next_level_for_children = current_nesting_level + 1
            else:
                next_level_for_children = current_nesting_level
            
            for child_node in node.get_children():
                traverse_ast(child_node, next_level_for_children)

        if tu.cursor: # Ensure cursor is valid before traversal
            traverse_ast(tu.cursor)
        
        features['control_count'] = _control_count
        features['max_control_depth'] = _max_depth
        
        features['num_funcs'] = len(_defined_funcs)
        func_lengths = []
        if _defined_funcs:
            for func_node in _defined_funcs:
                if func_node.extent and func_node.extent.start.line is not None and func_node.extent.end.line is not None:
                    # Line numbers are 1-based and inclusive
                    length = func_node.extent.end.line - func_node.extent.start.line + 1
                    if length >= 0: # Sanity check for valid length
                        func_lengths.append(length)
        
        if func_lengths: # Avoid division by zero if no valid function lengths were found
            features['avg_func_length'] = round(sum(func_lengths) / len(func_lengths), 2)
        else:
            features['avg_func_length'] = 0.0

        # 4) Token-based features (count and unique identifiers)
        tokens = []
        if tu.cursor and tu.cursor.extent and tu.cursor.extent.start.file: # Check valid extent
            try:
                tokens = list(tu.get_tokens(extent=tu.cursor.extent))
            except Exception as e:
                print(f"[WARNING] Could not get tokens for {file_path}: {e}. Token-based features will be zero.")
        
        features['token_count'] = len(tokens)
        
        unique_identifiers = set()
        if tokens:
            try:
                unique_identifiers = {token.spelling for token in tokens if token.kind == cindex.TokenKind.IDENTIFIER}
            except Exception as e:
                print(f"[WARNING] Error processing tokens for identifiers in {file_path}: {e}")
        features['unique_identifiers'] = len(unique_identifiers)

    except FileNotFoundError:
        print(f"[ERROR] Syntactic feature extraction: File not found at {file_path}")
        # features will retain their default zero values
    except Exception as e:
        print(f"[ERROR] Unexpected error during syntactic feature extraction for {file_path}: {e}")
        # features will retain their default zero values
    
    return features

# --- CodeBERT Embedding Extraction ---
def extract_codebert_embedding(code_text: str, max_length: int = 256):
    if TOKENIZER is None or MODEL is None:
        print("[ERROR] CodeBERT model/tokenizer not loaded. Cannot extract embeddings.")
        return [0.0] * 768 # Return a zero vector of the expected size

    try:
        encoded_input = TOKENIZER(
            code_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" # PyTorch tensors
        )
        input_ids = encoded_input["input_ids"].to(DEVICE)
        attention_mask = encoded_input["attention_mask"].to(DEVICE)

        with torch.no_grad(): # Disable gradient calculations for inference
            model_outputs = MODEL(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            last_hidden_states = model_outputs.last_hidden_state 
        
        # CLS token embedding is the first token's embedding in the sequence
        cls_embedding_tensor = last_hidden_states[0, 0, :] 
        return cls_embedding_tensor.cpu().numpy().tolist() # Convert to list of floats
    except Exception as e:
        print(f"[ERROR] Failed to extract CodeBERT embedding: {e}")
        return [0.0] * 768 # Return a zero vector on error

# --- Main Processing Function ---
def process_single_cpp_file(input_cpp_path: Path):
    if not input_cpp_path.exists():
        print(f"[ERROR] Input C++ file not found: {input_cpp_path}")
        return

    print(f"[INFO] Starting processing for file: {input_cpp_path}")

    # 1. Extract syntactic (Clang-based) features
    print("[INFO] Extracting syntactic features...")
    syntactic_features = extract_syntactic_features(input_cpp_path)
    # syntactic_features is guaranteed to be a dict, even if features are zeroed out on error.

    # 2. Read C++ file content for CodeBERT
    print("[INFO] Reading file content for CodeBERT embedding...")
    code_content = ""
    try:
        with open(input_cpp_path, "r", encoding="utf-8") as f:
            code_content = f.read()
    except UnicodeDecodeError: # Fallback to ISO-8859-1 if UTF-8 fails
        print("[WARNING] UTF-8 decoding failed. Trying ISO-8859-1...")
        try:
            with open(input_cpp_path, "r", encoding="ISO-8859-1") as f:
                code_content = f.read()
        except Exception as e_iso:
            print(f"[ERROR] Failed to read file {input_cpp_path} with UTF-8 or ISO-8859-1: {e_iso}")
            # CodeBERT embedding will likely be poor or fail for empty/unreadable content
    except Exception as e_read:
        print(f"[ERROR] Failed to read file {input_cpp_path}: {e_read}")

    # 3. Extract CodeBERT embeddings
    print("[INFO] Extracting CodeBERT embeddings...")
    codebert_embedding_vector = extract_codebert_embedding(code_content)
    # codebert_embedding_vector is a list of 768 floats (or zeros on error).

    # 4. Combine all features into a single dictionary
    combined_features = {}
    combined_features['filename'] = os.path.basename(input_cpp_path)
    
    # Add syntactic features
    for key, value in syntactic_features.items():
        combined_features[key] = value

    # Add code size
    combined_features['code_size'] = Path(input_cpp_path).stat().st_size

    # Add CodeBERT embedding features (vec_0, vec_1, ..., vec_767)
    for i, embedding_value in enumerate(codebert_embedding_vector):
        combined_features[f"vec_{i}"] = float(embedding_value)

    # 5. Write the combined features to a CSV file
    # Define fieldnames in a specific order for the CSV header
    # Ensure 'filename' is first, then syntactic features, then CodeBERT vectors.
    fieldnames = ['filename'] + \
             list(syntactic_features.keys()) + \
             ['code_size'] + \
             [f"vec_{i}" for i in range(len(codebert_embedding_vector))]
    
    # Make DataFrame
    pd_features = pd.DataFrame([combined_features])
    return pd_features

def extract_cpp_features_from_file(cpp_file_path: Path):
    # Load the CodeBERT model and tokenizer once at the start
    load_codebert_model() # This will exit if model loading fails

    # Process the single C++ file
    features = process_single_cpp_file(cpp_file_path)

    print("[INFO] All processing finished.")
    return features
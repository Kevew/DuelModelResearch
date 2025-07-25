import os
import re
import json

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
mathlib_path = os.path.join(parent_path, "mathlib4") 
output_dir = os.path.join(current_path, "unfiltered_dataset")

def consolidate_simpa_lines(text):
    """
    Consolidates multi-line `simpa only` calls into a single line.

    Example:
        simpa only [a, b,
                          c] at h1
        => simpa only [a,b,c] at h1
    """
    pattern = re.compile(
        r"^([ \t]*)(·\s+)?(simpa only)\s*\[([\s\S]+?)\](.*)$",
        re.MULTILINE
    )

    def repl(match):
        indent = match.group(1)
        bullet = match.group(2) or ""
        verb = match.group(3)
        content = match.group(4)
        rest = match.group(5)


        lemmas = [
            lemma.strip()
            for lemma in content.replace('\n', ' ').split(',')
            if lemma.strip()
        ]
        consolidated_content = ",".join(lemmas)

        return f"{indent}{bullet}{verb} [{consolidated_content}]{rest}"

    return pattern.sub(repl, text)


def split_simp_lines(text):
    """
    Splits multi-lemma `simp only` calls into separate lines.

    Example:
        simp only [a, b,
                   c]
        => simp only [a]\nsimp only [b]\nsimp only [c]
    """
    pattern = re.compile(
        r"^([ \t]*)(simp only)\s*\[([^\]]+)\](.*)$",
        re.MULTILINE
    )

    def repl(match):
        indent = match.group(1)
        # 'simp only'
        verb = match.group(2)
        content = match.group(3)
        rest = match.group(4)
        # Normalize lemmas list
        lemmas = [lemma.strip() for lemma in content.replace('\n', ' ').split(',') if lemma.strip()]
        # Reconstruct separate calls
        lines = []
        for lemma in lemmas:
            lines.append(f"{indent}{verb} [{lemma}]{rest}")
        return "\n".join(lines)

    return pattern.sub(repl, text)

def split_rw_lines(text):
    """
    Splits multi-lemma simp calls into separate lines.

    Example:
        rw [a, b,
              c]
        => rw [a]\nrw [b]\nrw [c]
    """
    pattern = re.compile(
        r"^([ \t]*)rw\s*\[([^\]]+)\](.*)$",
        re.MULTILINE
    )

    def repl(match):
        indent = match.group(1)
        content = match.group(2)
        rest = match.group(3)
        # Normalize lemmas list
        lemmas = [lemma.strip() for lemma in content.replace('\n', ' ').split(',') if lemma.strip()]
        # Reconstruct separate rw calls
        lines = []
        for lemma in lemmas:
            lines.append(f"{indent}rw [{lemma}]{rest}")
        return "\n".join(lines)

    return pattern.sub(repl, text)



def extract_declarations_from_file(file_path):
    """
    Extracts theorems, lemmas, and examples from a single Lean file.

    Args:
        file_path (str): The absolute path to the .lean file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              information about a found declaration.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    thereom_regex = re.compile(r"^\s*(private\s+|protected\s+)?(theorem|lemma|example)\s*(.*)", re.IGNORECASE)

    # Context stacks
    open_context = []  # sections and namespaces
    current_variables = []

    # Regexes for context
    open_regex = re.compile(r"^\s*(open)\s+(.+)", re.IGNORECASE)
    variable_regex = re.compile(r"^\s*(variable|variables)\b(.*)", re.IGNORECASE)



    # Don't find block lemma thereoms comments as they are not verfied by lean server
    block_doc_comment_start_regex = re.compile(r"^\s*/-")
    block_doc_comment_end_regex = re.compile(r"-/")

    in_block_doc_comment = False
    i = 0
    while i < len(lines):
        line = lines[i]

        line_content_stripped = line.strip()
        original_line_content = line.rstrip('\n')

        # Ignore stuff in comments
        if block_doc_comment_start_regex.match(line_content_stripped):
            in_block_doc_comment = True
            if block_doc_comment_end_regex.search(line_content_stripped) and not line_content_stripped.endswith("-!") : # Single line block comment e.g. /-! text !-/
                 in_block_doc_comment = False
            i += 1
            continue
        if in_block_doc_comment:
            if block_doc_comment_end_regex.search(line_content_stripped):
                in_block_doc_comment = False
            i += 1
            continue


        # Context Handling
        # Track open
        open_regex_val = open_regex.match(line_content_stripped)
        if open_regex_val:
            open_context.append(open_regex_val.group(2).strip())
            i += 1
            continue

        # Track variable declarations in context
        var_match = variable_regex.match(line_content_stripped)
        if var_match:
            # Append the full variable clause
            current_variables.append(var_match.group(2).strip())
            i += 1
            continue

        # Found something interesting
        match = thereom_regex.match(lines[i])
        if match:
            header = line.rstrip('\n')         # the matched header line
            decl_lines = [header]

            # measure the indentation of the header
            indent = len(line) - len(line.lstrip(' '))
            j = i + 1
            # grab everything more-indented (or blank) as part of the body
            while j < len(lines):
                next_line = lines[j]
                # blank lines belong to body
                if next_line.strip() == "":
                    decl_lines.append(next_line.rstrip('\n'))
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip(' '))
                if next_indent > indent:
                    decl_lines.append(next_line.rstrip('\n'))
                    j += 1
                else:
                    break

            declaration_content = "\n".join(decl_lines)
            declaration_content = consolidate_simpa_lines(declaration_content)
            declaration_content = split_simp_lines(declaration_content)
            declaration_content = split_rw_lines(declaration_content)

            # record the declaration
            data.append({
                "declaration": declaration_content,
                "file": file_path,
                "context": {
                    "open": list(open_context),
                    "variables": list(current_variables)
                }
            })

            # skip ahead
            i = j
            continue
            
        i += 1

    return data


def main():
    if not os.path.isdir(mathlib_path):
        print(f"Error: Path '{mathlib_path}' does not exist or is not a directory.")
        return
    
    print(f"\nScanning for .lean files in {mathlib_path}...")
    lean_files_to_scan = []
    for root, dirs, files in os.walk(mathlib_path):
        # Skip .lake directory (build artifacts and cache)
        if ".lake" in root.split(os.sep):
            continue
        # Skip .git directory
        if ".git" in root.split(os.sep):
            continue
        
        for file in files:
            if file.endswith(".lean"):
                lean_files_to_scan.append(os.path.join(root, file))

    total_files = len(lean_files_to_scan)
    print(f"Found {total_files} .lean files to scan.")

    all_theorems = []

    for idx, file_path in enumerate(lean_files_to_scan):
        if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
            relative_file_path = os.path.relpath(file_path, mathlib_path)
            print(f"Processing file ({idx + 1}/{total_files}): {relative_file_path}")

        thereoms = extract_declarations_from_file(file_path)
        all_theorems.extend(thereoms)

    print(f"\nExtraction complete. Found {len(all_theorems)} theorems, lemmas, and examples.")

    
    # Split the data into seperate ones
    chunk_size = 6000
    for idx in range(0, len(all_theorems), chunk_size):
        chunk = all_theorems[idx: idx + chunk_size]
        file_index = idx // chunk_size + 1
        out_path = os.path.join(output_dir, f"dataset_{file_index:03d}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as out:
            for decl in chunk:
                out.write(json.dumps(decl, ensure_ascii=True) + '\n')

    print("Finished")

if __name__ == "__main__":
    main()
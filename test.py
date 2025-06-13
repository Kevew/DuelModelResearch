print("Starting up!")
import json
import re
import os
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject, LeanRequire, LocalProject
import argparse
print("Loaded Imports")

PROGRESS_FILE = "data/progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def extract_traced_states(resp, lines):
    """Collects goal states from resp.infotree: all goalsBefore plus final goalsAfter, and captures "Goals accomplished!"."""
    states = []
    last_info = None
    tactic_idx = 0
    for info in getattr(resp, 'infotree', []):
        if info.get('kind') == 'TacticInfo':
            last_info = info
            after = last_info.get("node").get("goalsAfter")
            if tactic_idx == 0:
                # Inital one
                before = last_info.get("node").get("goalsBefore")
                state_text = "\n---\n".join(before)
                states.append(state_text)
            if before:
                state_text = "\n---\n".join(after)
                states.append(state_text)
                tactic_idx += 1
    return states

def get_tactic_states_from_lean_code(lean_code: str, project_path: str = "."):
    """
    Extracts the tactic state at each proof step using infotree.

    Send the entire proof in one command, then walk through infotree to record each intermediate goal.
    """
    if 'have?' in lean_code:
        print("Skipping proof containing 'have?'")
        return None
    lines = lean_code.splitlines()
    header_idx = None
    for i, l in enumerate(lines):
        stripped = l.strip()
        if re.match(r'^(theorem|lemma|def)\b', stripped):
            header_idx = i
            break

    if header_idx is None:
        header_idx = next((i for i, l in enumerate(lines) if re.match(r'^(theorem|lemma|def)\b', l.strip())), 0)

    file_ctx = lines[:header_idx]
    decl_header = lines[header_idx]
    body = [l for l in lines[header_idx+1:]]

    # Handle edge case: single-line "... := by " proofs
    edge_match = re.search(r':=\sby\s.*|:=\s.+', decl_header)
    if edge_match:
        if ':= by' in decl_header:
            sep = ':= by'
        else:
            sep = ':='
        before, after = decl_header.split(sep, 1)
        decl_header = before + sep
        body.insert(0, after.strip())

    # Build full proof string
    proof_lines = [decl_header]
    for line in body:
        proof_lines.append(line)
    full_proof = "\n".join(file_ctx + proof_lines)

    # Setup Lean server
    config = LeanREPLConfig(
        lean_version="v4.19.0",
        project=LocalProject("mathlib4"),
        verbose=False
    )
    server = LeanServer(config)
    # load context
    for ctx_line in file_ctx:
        server.run(Command(cmd=ctx_line))

    # send the entire proof
    print("sending proof")
    resp = server.run(Command(cmd=full_proof, root_goals=True, infotree="substantive"))
    print("received proof")
    # extract states
    try:
        traced = extract_traced_states(resp, body)
    except Exception as e:
        print(f"Skipping line due to error: {e}")
        return ["FAILURE"]

    # label states: 'initial' from sorries, then each tactic's before-state
    labels = ['(initial)'] + [l for l in body]
    states = []
    # If the first state isn't from infotree, use sorries
    for label, state in zip(labels, traced):
        states.append({"line": label, "tactic_state": state})
    
    server.kill()
    return states


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process a single Lean dataset file to extract tactic states."
    )
    parser.add_argument(
        'filename',
        help="Name of the file inside 'unfiltered_dataset/' to process (e.g. dataset_001.jsonl)"
    )
    args = parser.parse_args()

    target = args.filename

    all_files = os.listdir("data/unfiltered_dataset")
    if target not in all_files:
        print(f"Error: '{target}' not found in '{all_files}'. Available files: {all_files}")

    input_folder = "data/unfiltered_dataset"
    output_folder = "data/processed_dataset"
    os.makedirs(output_folder, exist_ok=True)
    in_path = os.path.join(input_folder, target)
    out_path = os.path.join(output_folder, target)


    progress = load_progress()
    file_progress = progress.get('file', {})
    
    print("Updating: ", target)

    count = 1
    with open(in_path, 'r') as infile, open(out_path, 'w') as outfile:
        try:
            for line in infile:
                print("Line " + str(count) + " is being evaluated")
                count += 1
                item = json.loads(line)
                print("Evaluting from: " +  item["file"])
                # assemble full Lean code with imports, opens, and variables
                full_code = "import Mathlib\n"
                for o in item['context'].get('open', []):
                    full_code += f"open {o}\n"
                for v in item['context'].get('variables', []):
                    full_code += f"variable {v}\n"
                full_code += item['declaration']

                # get traced states
                states = get_tactic_states_from_lean_code(full_code, project_path=".")
                if states == ["FAILURE"] or states is None:
                    continue
                item['tactic_states'] = states

                # write the enriched JSON line
                outfile.write(json.dumps(item) + "\n")
                file_progress[target] = count
                save_progress({'file': file_progress})
        except Exception as e:
            print(f"Error at line {count} in {target}: {e}")
            save_progress({'file': file_progress})
            raise

    print(f"Processed all files from '{input_folder}' into '{output_folder}'.")
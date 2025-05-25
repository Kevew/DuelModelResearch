import json
import re
import os
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject, LeanRequire

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
    lines = lean_code.splitlines()
    try:
        header_idx = next(i for i, l in enumerate(lines) if l.strip().endswith(':= by'))
    except StopIteration:
        header_idx = next(i for i, l in enumerate(lines) if ':=' in l)
    file_ctx = lines[:header_idx]
    decl_header = lines[header_idx]
    body = [l for l in lines[header_idx+1:]]

    # Build full proof string
    proof_lines = [decl_header]
    for line in body:
        proof_lines.append(line)
    full_proof = "\n".join(file_ctx + proof_lines)

    # Setup Lean server
    config = LeanREPLConfig(
        lean_version="v4.19.0",
        project=TempRequireProject([
            LeanRequire(name="mathlib", git="https://github.com/leanprover-community/mathlib4.git", rev="v4.19.0")
        ]),
        verbose=False
    )
    server = LeanServer(config)
    # load context
    for ctx_line in file_ctx:
        server.run(Command(cmd=ctx_line))

    # send the entire proof
    resp = server.run(Command(cmd=full_proof, root_goals=True, infotree="substantive"))
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
    input_folder = "unfiltered_dataset"
    output_folder = "processed_dataset"
    os.makedirs(output_folder, exist_ok=True)
    
    # iterate over each file in the input folder
    for fname in os.listdir(input_folder):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        print("Updating: ", fname)
        if fname != "dataset_001.jsonl":
            break

        count = 1
        with open(in_path, 'r') as infile, open(out_path, 'w') as outfile:
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
                if states == ["FAILURE"]:
                    continue
                item['tactic_states'] = states

                # write the enriched JSON line
                outfile.write(json.dumps(item) + "\n")

    print(f"Processed all files from '{input_folder}' into '{output_folder}'.")
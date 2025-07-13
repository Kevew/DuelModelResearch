import json
import re
import os
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject, LocalProject, ProofStep
from lean_interact.interface import LeanError
from typing import Any, Dict, List

import argparse
print("Loaded Imports")
import time

PROGRESS_FILE = "progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def extract_tactic_data(cmd_resp: Any) -> List[Dict[str, Any]]:
    """
    Traverses the CommandResponse.infotree to extract a dataset of tactics.
    """
    dataset: List[Dict[str, Any]] = []

    def recurse(tree: Any) -> None:
        tactic = getattr(tree.node.stx, 'pp', None)
        before = getattr(tree.node, 'goals_before', []) or []
        after = getattr(tree.node, 'goals_after', []) or []
        
        # Only add entries that have a tactic and a before state
        if tactic and before:
            dataset.append({
                'tactic': tactic,
                'before_state': before,
                'after_state': after
            })

        for child in getattr(tree, 'children', []):
            recurse(child)

    for root in getattr(cmd_resp, 'infotree', []):
        recurse(root)

    return dataset


SUCCESS_COUNT = 0


def get_tactic_states_from_lean_code(server: LeanServer, lean_code: str, initial_env):
    """
    Extracts the tactic state at each proof step using infotree.
    """
    # 1. Create a single command for the entire proof with infotree enabled
    cmd = Command(
        cmd=lean_code,
        env=initial_env,
        infotree="substantive"  # This flag is crucial
    )
    resp = server.run(cmd)

    # 2. Handle errors, including retrying on name clashes
    if isinstance(resp, LeanError):
        # Retry once if the theorem name already exists
        if "has already been declared" in str(resp.message):
            modified_code = re.sub(r"((?:theorem|lemma)\s+)(\S+)", r"\1\2_extra", lean_code, count=1)
            cmd_retry = Command(cmd=modified_code, env=initial_env, infotree="substantive")
            resp = server.run(cmd_retry)
            if isinstance(resp, LeanError):
                return resp, []
        else:
            return resp, []
            
    if not getattr(resp, 'infotree', None):
        return ["FAILURE"]

    # 3. Extract the tactic data from the infotree
    tactic_data = extract_tactic_data(resp)
    if not tactic_data:
        return ["FAILURE"], []

    # 4. Format the extracted data to match the script's expected output
    states = []
    # Add the initial goal state from the very first tactic encountered
    initial_goals = tactic_data[0]['before_state']
    states.append({"line": "(initial)", "tactic_state": "\n---\n".join(initial_goals)})

    # Process each subsequent tactic
    for step in tactic_data:
        tactic_text = step['tactic'].strip()
        after_state_text = "\n---\n".join(step['after_state'])

        if (after_state_text == ""):
            states.append({"line": tactic_text, "tactic_state": "No Goals!"})
            break
        
        states.append({"line": tactic_text, "tactic_state": after_state_text})
    
    if states[-1]["tactic_state"] != "No Goals!":
        return ["FAILURE"]

    global SUCCESS_COUNT
    SUCCESS_COUNT += 1
    print("Success")
    
    return states

def create_server(config):
    server = LeanServer(config)
    print("✅ Lean Server is ready.")
    print("Importing Mathlib for the session... This may take a moment.")
    mathlib_resp = server.run(Command(cmd="import Mathlib"))
    print(f"✅ Mathlib imported (env={mathlib_resp.env}).")
    return server, mathlib_resp.env


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

    all_files = os.listdir("unfiltered_dataset")
    if target not in all_files:
        print(f"Error: '{target}' not found in '{all_files}'. Available files: {all_files}")

    input_folder = "unfiltered_dataset"
    output_folder = "processed_dataset"
    os.makedirs(output_folder, exist_ok=True)
    in_path = os.path.join(input_folder, target)
    out_path = os.path.join(output_folder, target)

    config = LeanREPLConfig(
        lean_version="v4.19.0",
        project=LocalProject("/root/DuelModelResearch/mathlib4"),
        verbose=True
    )
    server, initial_env = create_server(config)


    progress = load_progress()
    file_progress = progress.get('file', {})
    last_done = file_progress.get(target, 0)
    
    print("Updating: ", target)

    count = 1
    try:
        with open(in_path, 'r') as infile, open(out_path, 'a') as outfile:
            for line in infile:
                count += 1
                if count <= last_done:
                    continue

                if (count - 1) % 100 == 0 and count > 1:
                    # I got a error earlier about the server being overloaded
                    # Should probably just dedicate more compute next time
                    print(f"\n--- Restarting Lean Server after 100 items ---\n")
                    server.kill()
                    server, initial_env = create_server(config)

                print(f"Line {count} is being evaluated")

                item = json.loads(line)
                start_time = time.time()

                full_code = "section\n"                 
                for o in item['context'].get('open', []):
                    full_code += f"open {o}\n"
                for v in item['context'].get('variables', []):
                    full_code += f"variable {v}\n"
                
                full_code += item['declaration']
                full_code += "\nend" # End the section

                states = get_tactic_states_from_lean_code(server, full_code, initial_env)

                end_time = time.time()
                duration = end_time - start_time
                print(f"  > Processed in {duration:.2f} seconds.") # Optional: for real-time feedback

                
                if states == ["FAILURE"] or states is None or isinstance(states, LeanError):
                    continue
                
                item['tactic_states'] = states
                outfile.write(json.dumps(item) + "\n")

                file_progress[target] = count
                save_progress({'file': file_progress})

    except Exception as e:
        print(f"An error occurred at line {count} in {target}: {e}")
        save_progress({'file': file_progress})
        raise
    finally:
        save_progress({'file': file_progress})
        print("Shutting down Lean Server.")
        server.kill()
        print("Succesfully generated: " + str(SUCCESS_COUNT))


    print(f"Processed all files from '{input_folder}' into '{output_folder}'.")
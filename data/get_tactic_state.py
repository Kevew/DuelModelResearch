print("Starting up!")
import json
import re
import os
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject, LocalProject, ProofStep
from lean_interact.interface import LeanError

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

'''
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
'''

def get_tactic_states_from_lean_code(server: LeanServer, lean_code: str, initial_env):
    """
    Extracts the tactic state at each proof step using infotree.

    Send the entire proof in one command, then walk through infotree to record each intermediate goal.
    """
    if 'have?' in lean_code:
        print("Skipping proof containing 'have?'")
        return None
    # The incoming lean_code is wrapped in "section...end", so we can ignore them for parsing.
    inner_code = "\n".join(lean_code.splitlines()[1:-1])

    # Find where the proof body starts (after `:= by`)
    match = re.search(r':=\s*by\b', inner_code)
    if not match:
        print(f"  > Error: Could not find ':= by' in the declaration.")
        return ["FAILURE"]

    header_end_pos = match.start() + 2
    header = inner_code[:header_end_pos].strip()
    body_str = inner_code[match.end():]
    
    # Filter out empty lines and comments from the proof body
    proof_lines = []
    for raw in body_str.splitlines():
        txt = raw.strip()
        if not txt or txt.startswith('--'):
            continue
        if txt.startswith("by "):
            txt = txt[len("by "):]
        proof_lines.append(txt)
    states = []
    initialization_code = f"section\n{header} sorry"
    print(f"  > Sending header to get initial state...")
    resp = server.run(Command(cmd=initialization_code, root_goals=True, env=initial_env))
    
    # Since this is in mathlib, sometimes this will find the theorem already declared so we gotta account for that
    if any(
        msg.severity == 'error' and "has already been declared" in msg.data
        for msg in getattr(resp, "messages", [])
    ):
        header = re.sub(r"(theorem\s+)(\S+)", r"\1\2_extra", header, count=1)
        initialization_code = f"section\n{header} sorry"
        print(" > Name collision: retrying as: ")
        resp = server.run(Command(cmd=initialization_code, root_goals=True, env=initial_env))

    initial_goals = []
    initial_sorries = getattr(resp, 'sorries', [])
    if initial_sorries:
        initial_goals = [s.goal for s in initial_sorries]
    else:
        initial_goals = getattr(resp, 'goalsAfter', [])
    print(initialization_code)
    if not initial_goals:
        # Check if the goal was solved immediately
        if any("Goals accomplished!" in log.get('msg', '') for log in getattr(resp, 'log', [])):
             initial_state_text = "Goals accomplished!"
             states.append({"line": "(initial)", "tactic_state": initial_state_text})
             if proof_lines:
                states.append({"line": proof_lines[0], "tactic_state": "Goals accomplished!"})
             return states
        else:
            print("  > Warning: No initial goals found. There might be an error in the declaration.")
            return ["FAILURE"]
    initial_state_text = "\n---\n".join(initial_goals)
    states.append({"line": "(initial)", "tactic_state": initial_state_text})

    proof_state_id = getattr(resp.sorries[0], 'proof_state', [])
    for tactic_line in proof_lines:
        print(tactic_line)
        resp = server.run(ProofStep(tactic=tactic_line, proof_state=proof_state_id))
        # For some reason, stuff like lift aint working
        if isinstance(resp, LeanError):
            return ["FAILURE"]
        proof_state_id = resp.proof_state
        
        current_goals = getattr(resp, 'goals', [])
        state_text = ""

        if resp.proof_status == "Incomplete: open goals remain":
            state_text = "\n---\n".join(current_goals)
        else:
            state_text = "No Goals!"

        states.append({"line": tactic_line, "tactic_state": state_text})
        
        # If the goal is solved, we can stop early.
        if state_text == "Goals accomplished!":
            break

    # --- 4. Close the section to keep the environment clean ---
    server.run(Command(cmd="end"))

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
        project=LocalProject("/home/kevew/scratch/DuelModelResearch/mathlib4"),
        verbose=True
    )
    server, initial_env = create_server(config)


    progress = load_progress()
    file_progress = progress.get('file', {})
    
    print("Updating: ", target)

    count = 1
    try:
        with open(in_path, 'r') as infile, open(out_path, 'w') as outfile:
            for line in infile:
                print(f"Line {count} is being evaluated")
                count += 1

                if (count - 1) % 100 == 0 and count > 1:
                    print(f"\n--- Restarting Lean Server after 100 items ---\n")
                    server.kill()
                    server, initial_env = create_server(config)

                item = json.loads(line)
                print("Evaluating from: " + item["file"])
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

                
                if states == ["FAILURE"] or states is None:
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
        print("Shutting down Lean Server.")
        server.kill()


    print(f"Processed all files from '{input_folder}' into '{output_folder}'.")
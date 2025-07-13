# Ignore this file, was doing some testing

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

def tactic_state_to_theorem(tactic_state: str, theorem_name: str = "my_theorem") -> str:
    """
    Converts a Lean 4 tactic state string into a Lean 4 theorem.

    Args:
        tactic_state: A string representing the Lean 4 tactic state.
        theorem_name: The desired name for the generated theorem.

    Returns:
        A string representing the Lean 4 theorem.
    """
    # Split the tactic state into lines
    lines = tactic_state.strip().split('\n')

    # The last line is the goal, everything before is hypotheses
    goal = lines[-1].replace("⊢ ", "").strip()
    hypotheses = lines[:-1]

    # Process hypotheses to extract variable names and types
    processed_hypotheses = []
    for h in hypotheses:
        if ":" in h:
            # Regular expression to handle potential dependencies in hypotheses
            match = re.match(r"(\w+)\s*:\s*(.*)", h.strip())
            if match:
                var_name = match.group(1)
                var_type = match.group(2).strip()
                processed_hypotheses.append(f"({var_name} : {var_type})")

    # Construct the theorem signature
    theorem_signature = f"theorem {theorem_name} " + " ".join(processed_hypotheses) + f" : {goal} :="

    # Add a placeholder for the proof
    proof = "  sorry"

    # Combine to form the final theorem
    return f"{theorem_signature}\n{proof}"

def generate_theorem_from_have(tactic_state: str, theorem_name: str, have_decl: str, have_proof: str) -> str:
    """
    Generates a complete theorem from a 'have' statement and the current tactic state.
    
    Args:
        tactic_state: The tactic state providing the hypotheses.
        theorem_name: The name for the new theorem.
        have_decl: The declaration part of the have (e.g., 'h : P').
        have_proof: The proof part of the have (e.g., 'by assumption').

    Returns:
        A string representing the complete Lean 4 theorem.
    """
    # The goal of our new theorem is the proposition from the 'have' statement.
    goal = have_decl.split(':', 1)[-1].strip()
    
    # Reconstruct the tactic state with the correct goal for the 'have' proposition.
    lines = tactic_state.strip().split('\n')
    hypotheses = [line for line in lines if not line.startswith('⊢')]
    new_tactic_state = '\n'.join(hypotheses) + f'\n⊢ {goal}'

    # Use the existing function to create the theorem shell (header + signature).
    theorem_shell = tactic_state_to_theorem(new_tactic_state, theorem_name)

    # Replace the 'sorry' with the actual proof from the 'have' statement.
    proof_body = have_proof.strip()
    if not proof_body.startswith("by"):
        proof_body = f"by {proof_body}"
    
    final_theorem = theorem_shell.replace("  sorry", f"  {proof_body}")
    return final_theorem


def group_proof_steps(body_str: str):
    """
    Groups lines of a proof body into distinct tactic steps based on indentation.
    This is a heuristic and may not be perfect for all Lean syntax.
    """
    steps = []
    lines = [line for line in body_str.splitlines() if line.strip() and not line.strip().startswith('--')]
    if not lines:
        return []

    current_step = lines[0]
    base_indent = len(lines[0]) - len(lines[0].lstrip(' '))

    for i in range(1, len(lines)):
        line = lines[i]
        indent = len(line) - len(line.lstrip(' '))
        # If indentation increases, it's part of the current tactic.
        # If it's the same or less, it's a new tactic.
        if indent > base_indent:
            current_step += "\n" + line
        else:
            steps.append(current_step.strip())
            current_step = line
            base_indent = indent
    
    steps.append(current_step.strip())
    return steps

SUCCESS_COUNT = 0


def get_tactic_states_from_lean_code(server: LeanServer, lean_code: str, initial_env):
    """
    Extracts the tactic state at each proof step using infotree.

    Send the entire proof in one command, then walk through infotree to record each intermediate goal.
    """
    if 'have?' in lean_code:
        print("Skipping proof containing 'have?'")
        return None, []
    newly_generated_theorems = []
    # The incoming lean_code is wrapped in "section...end", so we can ignore them for parsing.
    inner_code = "\n".join(lean_code.splitlines()[1:-1])
    match_hdr = re.search(r"\b(?:theorem|lemma)\s+(\w+)", inner_code)
    theorem_name = match_hdr.group(1) if match_hdr else "my_theorem"


    # Find where the proof body starts (after `:= by`)
    match = re.search(r':=\s*by\b', inner_code)
    if not match:
        print(f"  > Error: Could not find ':= by' in the declaration.")
        return ["FAILURE"], []


    header_end_pos = match.start() + 2
    header = inner_code[:header_end_pos].strip()
    body_str = inner_code[match.end():]
    
    
    proof_steps = group_proof_steps(body_str)

    states = []
    initialization_code = f"section\n{header} sorry"
    resp = server.run(Command(cmd=initialization_code, root_goals=True, env=initial_env))
    # Since this is in mathlib, sometimes this will find the theorem already declared so we gotta account for that
    if any(msg.severity == 'error' and "has already been declared" in msg.data
        for msg in getattr(resp, "messages", [])):
        header = re.sub(r"((?:theorem|lemma)\s+)(\S+)", r"\1\2_extra", header, count=1)
        initialization_code = f"section\n{header} sorry"
        resp = server.run(Command(cmd=initialization_code, root_goals=True, env=initial_env))

    print(initialization_code)
    initial_goals = getattr(resp, 'sorries', [])[0].goal if getattr(resp, 'sorries', []) else getattr(resp, 'goalsAfter', [])
    if not initial_goals:
        print("  > Warning: No initial goals found.")
        return ["FAILURE"], []

    states.append({"line": "(initial)", "tactic_state": initial_goals})

    proof_state_id = getattr(resp.sorries[0], 'proof_state', None)
    if not proof_state_id:
        return ["FAILURE"], []

    current_tactic_state = initial_goals

    for tactic_line in proof_steps:
        tactic = tactic_line
        match_have = re.match(
            r"^have\s+(?:(\w+)\s*:\s*)?([^=]+)\s*:=\s*by\b(.*)",
            tactic.strip(),
            re.DOTALL
        )
        if match_have:
            have_name_group = match_have.group(1)
            have_type = match_have.group(2).strip()
            have_proof = match_have.group(3).strip()
            # Build declaration, handling optional name
            if have_name_group:
                have_decl = f"{have_name_group} : {have_type}"
            else:
                have_decl = f": {have_type}"
            new_theorem_name = f"{theorem_name}_have_{len(newly_generated_theorems) + 1}"
            new_theorem_code = generate_theorem_from_have(current_tactic_state, new_theorem_name, have_decl, have_proof)
            newly_generated_theorems.append({'declaration': new_theorem_code})
            tactic = f"have {have_decl} := sorry"
        print(tactic)
        resp = server.run(ProofStep(tactic=tactic, proof_state=proof_state_id))
        # For some reason, stuff like lift aint working
        if isinstance(resp, LeanError):
            return resp, []
        proof_state_id = resp.proof_state
        
        proof_state_id = resp.proof_state
        state_text = ""
        if resp.proof_status == "Incomplete: open goals remain":
            state_text = "\n---\n".join(resp.goals)
        else:
            state_text = "No Goals!"
        
        current_tactic_state = state_text
        states.append({"line": tactic, "tactic_state": state_text})
    global SUCCESS_COUNT
    SUCCESS_COUNT += 1
    print("Success")
    server.run(Command(cmd="end"))
    return states, newly_generated_theorems


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

                states, new_theorems = get_tactic_states_from_lean_code(server, full_code, initial_env)

                end_time = time.time()
                duration = end_time - start_time
                print(f"  > Processed in {duration:.2f} seconds.") # Optional: for real-time feedback

                
                if states == ["FAILURE"] or states is None or isinstance(states, LeanError):
                    if isinstance(states, LeanError):
                        with open("fail.txt", "a") as myfile:
                            myfile.write(str(states) + f" - Line {count}" + '\n')
                    continue
                
                item['tactic_states'] = states
                outfile.write(json.dumps(item) + "\n")

                # Write the newly generated theorems from 'have' statements to the output
                for new_theorem_item in new_theorems:
                    # These new items inherit the context of the parent theorem
                    new_item_to_write = {'context': item['context'], 'declaration': new_theorem_item['declaration']}
                    with open("extra.txt", "a") as myfile:
                        myfile.write(json.dumps(new_item_to_write) + "\n")
                    print(f"  > Generated and saved new theorem from 'have': {new_theorem_item['declaration'].split(':=', 1)[0]}")

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
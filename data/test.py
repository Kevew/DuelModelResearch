from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject
from lean_interact.interface import LeanError

import json
from typing import Any, Dict, List

def create_server(config):
    server = LeanServer(config)
    print("✅ Lean Server is ready.")
    print("Importing Mathlib for the session... This may take a moment.")
    mathlib_resp = server.run(Command(cmd="import Mathlib"))
    print(f"✅ Mathlib imported (env={mathlib_resp.env}).")
    return server, mathlib_resp.env

config = LeanREPLConfig(project=TempRequireProject(require="mathlib"))

server, initial_env = create_server(config)

initialization_code = """
theorem amc12a_2008_p8 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y)
  (h₁ : y^3 = 1)
  (h₂ : 6 * x^2 = 2 * (6 * y^2)) :
    x^3 = 2 * Real.sqrt 2 := by {
  have h1: y = 1 := by {
    have h3 : y^3 - 1 = 0 := by linarith
    have h4 : y^3 - 1 = (y - 1) * (y^2 + y + 1) := by ring
    rw [h4] at h3
    cases (mul_eq_zero.1 h3) with
    | inl h5 =>
      linarith
    | inr h6 =>
      have h7 : y^2 + y + 1 > 0 := by
        nlinarith [sq_nonneg (y + 1 / 2)]
      nlinarith
  }
  rw [h1] at h₂
  have h4 : x^2 = 2 := by linarith
  have hx : x = Real.sqrt 2 := by {
    have h6 : x^2 = 2 := h4
    rw [←h6]
    have h5 : x > 0 := h₀.left
    field_simp
  }
  rw [hx]
  have h8 : (Real.sqrt 2) ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  nlinarith
}
"""

cmd = Command(
    cmd=initialization_code,
    env=initial_env,
    infotree="substantive"
)
def extract_tactic_data(cmd_resp: Any) -> List[Dict[str, Any]]:
    """
    Traverses the CommandResponse.infotree to extract a dataset of tactics.

    Each entry in the returned list is a dict with:
      - 'tactic': the tactic source text (pp)
      - 'before_state': list of goal states before the tactic
      - 'after_state': list of goal states after the tactic
    """
    dataset: List[Dict[str, Any]] = []

    def recurse(tree: Any) -> None:
        # Extract the tactic text
        tactic = getattr(tree.node.stx, 'pp', None)
        # Extract goal states
        before = getattr(tree.node, 'goals_before', []) or []
        after = getattr(tree.node, 'goals_after', []) or []

        # Append to dataset
        dataset.append({
            'tactic': tactic,
            'before_state': before,
            'after_state': after
        })

        # Recurse into children
        for child in getattr(tree, 'children', []):
            recurse(child)

    # Start traversal for each root in infotree
    for root in getattr(cmd_resp, 'infotree', []):
        recurse(root)

    dataset.pop(0)
    return dataset

resp = server.run(cmd)
print(resp)
print(extract_tactic_data(resp))
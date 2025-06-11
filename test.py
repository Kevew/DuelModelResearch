import os
import subprocess
import json

current_path = os.path.dirname(os.path.abspath(__file__))
mathlib_path = os.path.join(current_path, "mathlib4")
file_path = os.path.join(mathlib_path, "test.lean")


print("Checking path: " + file_path)
print("Running")
result = subprocess.run(
    ["lake", "env", "lean", "--run", file_path, "--json"],
    capture_output=True,
    text=True,
    check=False
)

info_objects = []
print(result)
for line in result.stdout.splitlines():
    try:
        info_object = json.loads(line)
        info_objects.append(info_object)
    except json.JSONDecodeError:
        pass

print(info_objects)
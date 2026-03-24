import os
import re

openclay_dir = r"f:\SecurePrompt\openclay\openclay"

def replace_in_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content
    # Case-preserving replacements
    new_content = re.sub(r"PromptShield", "OpenClay", new_content)
    new_content = re.sub(r"promptshield", "openclay", new_content)
    new_content = re.sub(r"PROMPTSHIELD", "OPENCLAY", new_content)

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated: {path}")

for root, _, files in os.walk(openclay_dir):
    for filename in files:
        if filename.endswith(".py") or filename.endswith(".json") or filename.endswith(".yml") or filename.endswith(".yaml"):
            path = os.path.join(root, filename)
            replace_in_file(path)

print("Replacement complete.")

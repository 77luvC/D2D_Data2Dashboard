import json
import os

# Path to the notebook
notebook_path = 'notebooks/exp01_domain_stim_harnow_eval.ipynb'

# Load the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get the source code from the first cell
source_lines = notebook['cells'][0]['source']

# Find the line with the reference error and replace it
for i, line in enumerate(source_lines):
    if 'reference = ref_df.iloc[i]["reference"]' in line:
        source_lines[i] = '    # Print columns to debug\n    print("Available columns:", ref_df.columns.tolist())\n    # Use first column instead of reference column\n    reference = ref_df.iloc[i][ref_df.columns[0]]  # Using first column\n'
        break

# Save the modified notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook updated successfully!") 
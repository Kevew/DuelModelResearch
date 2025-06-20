import os
import glob

input_folder = 'processed_dataset'
output_file = 'combined_dataset.jsonl'

# Use glob to get all matching jsonl files
input_files = sorted(glob.glob(os.path.join(input_folder, 'dataset_0**.jsonl')))

with open(output_file, 'w', encoding='utf-8') as outfile:
    for fname in input_files:
        with open(fname, 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)

print(f"Combined {len(input_files)} files into {output_file}")

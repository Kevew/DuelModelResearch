import json

def process_jsonl(input_path, output_path):
    """
    Reads a JSONL file, processes each JSON object to the specified format,
    and writes the resulting strings to an output file.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            obj = json.loads(line)
            
            decl = obj.get('declaration', '')
            
            decl_sorry = decl.split(':=')[0].strip() + ' := by sorry'
            
            summary = obj.get('summary', '').strip()
                        
            # Format the final string
            formatted = (
                "Reason and complete the following lean4 proof:\n"
                f"{decl_sorry}\n"
                f"{summary}\n"
                f"{decl}"
            )

            wrapped = f"<ex>\n{formatted}\n</ex>"
            
            # Write to output file
            outfile.write(wrapped + "\n\n")

if __name__ == "__main__":
    input_file = "data.jsonl"    # Path to your input JSONL file
    output_file = "output.txt"   # Path where the formatted output will be saved
    process_jsonl(input_file, output_file)
    print(f"Processed JSONL data and saved formatted strings to '{output_file}'.")

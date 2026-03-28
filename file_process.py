
def process_file(input_filename, output_filename, max_lines=100):
    line_count = 0
    
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Remove line numbers at the beginning of each line
            cleaned_line = line
            
            # Find the first non-whitespace, non-digit, non-dash, non-dot character position
            i = 0
            while i < len(cleaned_line) and (cleaned_line[i].isdigit() or cleaned_line[i].isspace() or 
                                           cleaned_line[i] == '-' or cleaned_line[i] == '.'):
                i += 1
                
            # Extract the actual content after the numbering
            if i < len(cleaned_line):
                cleaned_line = cleaned_line[i:]
                
                # Write the cleaned line to the output file
                outfile.write(cleaned_line.strip() + '\n')
                
                line_count += 1
                if line_count >= max_lines:
                    break

if __name__ == "__main__":
    input_file = "ara-sa_newscrawl-OSIAN_2018_10K-sentences.txt"  # Change this to your input file name
    output_file = "sentences.txt"  # Change this to your desired output file name
    
    process_file(input_file, output_file)
    print(f"Processing complete. Output written to {output_file}")
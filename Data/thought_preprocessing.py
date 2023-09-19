import re

def process_file(file_name):
    line_num = 0
    with open(file_name, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        print(f'Processing line: {line_num}')
        # Remove empty lines
        if line.strip() == '':
            continue
        # Replace website links with '[website]'
        if 'https://' in line:
            line = re.sub(r'https?://[^\s]*', '(website)', line)
        # Remove lines containing only a "
        if line.strip() == '"':
            continue
        # Remove lines beginning with a date followed by a comma
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4},", line):
            continue
        processed_lines.append(line)
        line_num += 1

    with open('all_thoughts_cleaned.txt', 'w') as f:
        for line in processed_lines:
            f.write(line)

# Call the function with the path to your file
process_file('all_thoughts.txt')

import json
import argparse
import os
from transformers import AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LookAhead Tuning"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the input file (JSON or JSONL).'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to the output file (will be JSON or JSONL based on input_file).'
    )
    parser.add_argument(
        '--input_format',
        type=str,
        choices=['json', 'jsonl'],
        required=True,
        help='Format of the input file: "json" or "jsonl".'
    )
    parser.add_argument(
        '--output_format',
        type=str,
        choices=['json', 'jsonl'],
        required=True,
        help='Format of the output file: "json" or "jsonl".'
    )
    parser.add_argument(
        '--input_field',
        type=str,
        default='input',
        help='Field name for the input in the JSON objects.'
    )
    parser.add_argument(
        '--output_field',
        type=str,
        default='output',
        help='Field name for the output in the JSON objects.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['real', 'virtual'],
        required=True,
        help='Processing mode: "real" or "virtual".'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        help='Path to the model tokenizer (required for real mode).'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=6,
        help='Number of tokens to preview (required for real mode).'
    )
    parser.add_argument(
        '--P',
        type=str,
        default="Let's solve this problem. ",
        help='Predefined string to use in virtual mode (required for virtual mode).'
    )
    parser.add_argument(
        '--connector',
        type=str,
        default=" The answer begins with: ",
        help='String used to connect parts.'
    )
    return parser.parse_args()

def process_item_real(item, input_field, output_field, m, connector, tokenizer):
    """
    Real method: Concatenate the first m tokens from the output field to the input field.
    """
    # Tokenize the output field
    output_tokens = tokenizer.tokenize(item[output_field])
    first_m_tokens = output_tokens[:m]
    # Decode tokens back to string
    first_m_tokens_str = tokenizer.convert_tokens_to_string(first_m_tokens)
    # Concatenate input with connector and extracted tokens
    item[input_field] = item[input_field] + connector + first_m_tokens_str
    return item

def process_item_virtual(item, input_field, output_field, P, connector):
    """
    Virtual method: Concatenate the connector and P to the input field, and prepend P to the output field.
    """
    # Concatenate input with connector and P
    item[input_field] = item[input_field] + connector + P
    # Prepend P to the output field
    item[output_field] = P + item[output_field]
    return item

def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    return data

def read_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, 1):
            if line.strip():  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON decode error at line {line_number}: {e}")
    return data

def write_json(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

def write_jsonl(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in data:
            json_line = json.dumps(item)
            outfile.write(json_line + '\n')

def main():
    args = parse_arguments()
    
    if args.input_format == 'json':
        try:
            data = read_json(args.input_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON file '{args.input_file}': {e}")
    elif args.input_format == 'jsonl':
        try:
            data = read_jsonl(args.input_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read JSONL file '{args.input_file}': {e}")
    else:
        raise ValueError("Unsupported input file format. Please provide a .json or .jsonl file.")
    # Select processing method
    if args.mode == 'real':
        # Initialize tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from '{args.tokenizer_path}': {e}")
        if args.m is None:
            raise ValueError("Parameter 'm' is required for real mode.")
    elif args.mode == 'virtual':
        if args.P is None:
            raise ValueError("Parameter 'P' is required for virtual mode.")
    else:
        raise ValueError("Invalid mode. Please select 'real' or 'virtual'.")

    # Process each item
    processed_data = []
    for index, item in enumerate(data, 1):
        # Check if both input_field and output_field exist
        if args.input_field not in item:
            raise ValueError(f"Item {index} is missing the input_field '{args.input_field}'.")
        if args.output_field not in item:
            raise ValueError(f"Item {index} is missing the output_field '{args.output_field}'.")
        if args.mode == 'real':
            processed_item = process_item_real(
                item, 
                args.input_field, 
                args.output_field, 
                args.m, 
                args.connector,
                tokenizer
            )
        else:  # virtual
            processed_item = process_item_virtual(
                item, 
                args.input_field, 
                args.output_field, 
                args.P, 
                args.connector,
            )
        processed_data.append(processed_item)
        # Optionally, print progress for large datasets
        if index % 1000 == 0:
            print(f"Processed {index} items...")

    # Determine output format based on extension
    try:
        if args.output_format == 'jsonl':
            write_jsonl(args.output_file, processed_data)
        elif args.output_format == 'json':
            write_json(args.output_file, processed_data)
        else:
            raise ValueError("Unsupported output format. Please select 'json' or 'jsonl'.")
    except Exception as e:
        raise RuntimeError(f"Failed to write output file '{args.output_file}': {e}")
    print(f"Processing complete. Output saved to '{args.output_file}' in {args.output_format.upper()} format.")

if __name__ == "__main__":
    main()

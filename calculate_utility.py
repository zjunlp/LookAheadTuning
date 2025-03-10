import json
import argparse
from rouge_score import rouge_scorer


def calculate_rouge1_score(label, predict):
    """
    Compute the ROUGE-1 F1 score for a single sample.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_scores = scorer.score(label, predict)
    return rouge_scores['rouge1'].fmeasure


def rouge1_cal(file_path):
    """
    Parse and process the JSONL file, compute the ROUGE-1 score for each sample,
    and print the last element from the "metrics" field (if available) with multiplication by 100.
    """
    total_rouge1_score = 0.0
    count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Print the last element from metrics if it exists and multiply numeric values by 100
                if 'metrics' in data and isinstance(data['metrics'], list) and data['metrics']:
                    metric_val = data['metrics'][-1]
                    print("his ans:", metric_val)

                if 'results' not in data:
                    continue
                results = data['results']
                for x in results:
                    label = x.get('ground_truth', '')
                    predict = x.get('completion', '')
                    rouge1_score = calculate_rouge1_score(label, predict)
                    total_rouge1_score += rouge1_score
                    count += 1

        average_rouge1_score = total_rouge1_score / count if count > 0 else 0
        # Multiply scores by 100 and format to 2 decimal places
        print("Total ROUGE-1 Score: {:.2f}".format(total_rouge1_score * 100))
        print("Sample Count:", count)
        print("Average ROUGE-1 Score: {:.2f}".format(average_rouge1_score * 100))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def extract_final_number(text):
    """
    Extract the final number after the last occurrence of '####' in the text.
    Whitespace and commas are removed. Returns None if '####' is not found.
    """
    if '####' in text:
        return text.split('####')[-1].strip().replace(",", "")
    return None


def gsm8k_cal(file_path):
    """
    Parse and process the JSONL file (or file with one JSON object per line containing "results"),
    compare the final number extracted from the prediction text with the ground_truth,
    and calculate the accuracy. The resulting accuracy will be multiplied by 100 and formatted to 2 decimal places.
    """
    total_count = 0
    correct_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                # Print formatted metrics if available (all numeric values multiplied by 100)
                if 'metrics' in data:
                    metrics_list = data['metrics']
                    print("metrics:", metrics_list)

                if 'results' not in data:
                    continue
                results = data['results']
                for x in results:
                    total_count += 1
                    label_result = x.get('ground_truth', '').strip()
                    predict_text = x.get('completion', '')
                    predict_result = extract_final_number(predict_text)
                    if label_result == predict_result:
                        correct_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0
        # Multiply accuracy by 100 and format to 2 decimals
        print("Total Samples:", total_count)
        print("Correct Predictions:", correct_count)
        print("Accuracy: {:.2f}".format(accuracy * 100))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation tool: Supports ROUGE-1 scoring and GSM8K prediction accuracy measurement. "
                    "All computed percentage values are multiplied by 100 and displayed with 2 decimal places.")
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Choose evaluation mode: 'rouge1' or 'gsm8k'")

    # Subcommand for calculating ROUGE-1 score
    parser_rouge = subparsers.add_parser("rouge1", help="Compute ROUGE-1 score")
    parser_rouge.add_argument("--input_file", type=str, required=True,
                              help="Input JSONL file path (should contain 'metrics' and 'results' fields)")

    # Subcommand for GSM8K prediction accuracy
    parser_gsm8k = subparsers.add_parser("gsm8k", help="Compute GSM8K prediction accuracy")
    parser_gsm8k.add_argument("--input_file", type=str, required=True,
                              help="Input JSONL file path (should contain 'metrics' and 'results' fields)")

    args = parser.parse_args()

    if args.mode == "rouge1":
        print("Starting ROUGE-1 score computation...")
        rouge1_cal(args.input_file)
    elif args.mode == "gsm8k":
        print("Starting GSM8K prediction accuracy computation...")
        gsm8k_cal(args.input_file)
    else:
        parser.error("Please choose a valid mode: 'rouge1' or 'gsm8k'")


if __name__ == "__main__":
    main()
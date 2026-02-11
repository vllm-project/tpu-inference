import requests
import json
import random
import string
import argparse
import logging

def generate_prompt(num_words):
    """Generates a random prompt with a specified number of words."""
    words = [random.choice(string.ascii_lowercase) for _ in range(num_words)]
    return " ".join(words)

def send_request(url, prompt, model_name, output_length):
    """Sends a request to the LLM and returns the response."""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": output_length,
        "temperature": 0.0,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        # Assuming the response format is similar to OpenAI's
        return response.json()["choices"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Unexpected response format - {e}"


def main(args):
    """Main function to run the correctness test."""
    mismatches = 0
    for i in range(args.num_requests):
        logging.info(f"Sending request {i+1}/{args.num_requests}...")
        prompt = generate_prompt(args.input_length)

        baseline_response = send_request(args.baseline_url, prompt, args.model, args.output_length)
        disagg_response = send_request(args.disagg_url, prompt, args.model, args.output_length)

        if baseline_response != disagg_response:
            mismatches += 1
            logging.info(f"  MISMATCH FOUND for prompt {i+1}:")
            logging.info(f"    Prompt: {prompt[:100]}...")
            logging.info(f"    Baseline: {baseline_response}")
            logging.info(f"    Disagg:   {disagg_response}")
        else:
            logging.info(f"  Responses match for prompt {i+1}.")

    logging.info("\n--- Test Summary ---")
    if mismatches == 0:
        logging.info("All responses matched! The services are consistent.")
    else:
        logging.info(f"{mismatches}/{args.num_requests} requests had mismatched responses.")
        if mismatches > args.num_requests * 0.5:  # If more than 50% mismatches, raise an exception
            raise Exception("More than 50% of responses mismatched. There may be a significant issue.")
    logging.info("--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a correctness test between two LLM services.")
    parser.add_argument("--num_requests", type=int, default=20,
                        help="Number of requests to send.")
    parser.add_argument("--input_length", type=int, default=100,
                        help="Length of each input prompt in words.")
    parser.add_argument("--output_length", type=int, default=20,
                        help="Length of each output response in words.")
    parser.add_argument("--baseline_url", type=str, default="http://localhost:9400/v1/completions",
                        help="URL of the baseline LLM service.")
    parser.add_argument("--disagg_url", type=str, default="http://localhost:8000/v1/completions",
                        help="URL of the disaggregated LLM service.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Name of the model to use for the requests.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)

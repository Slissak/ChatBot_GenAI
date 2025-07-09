import os
import logging
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def upload_training_file(client, file_path):
    try:
        with open(file_path, "rb") as f:
            file = client.files.create(file=f, purpose="fine-tune")
        logging.info(f"Uploaded file. File ID: {file.id}")
        return file.id
    except Exception as e:
        logging.error(f"Failed to upload training file: {e}")
        raise


def create_fine_tune_job(client, model_name, training_file_id):
    try:
        fine_tune = client.fine_tuning.jobs.create(
            model=model_name,
            training_file=training_file_id,
            method={"type": "supervised"}
        )
        logging.info(f"Created fine-tune job. Job ID: {fine_tune.id}")
        print(f"Fine-tune job started. Job ID: {fine_tune.id}")
        return fine_tune.id
    except Exception as e:
        logging.error(f"Failed to create fine-tune job: {e}")
        raise


def poll_fine_tune_job(client, job_id, poll_interval=30):
    print(f"Polling fine-tune job status every {poll_interval} seconds...")
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            print(f"Job {job_id} status: {status}")
            if status == 'succeeded':
                print(f"Fine-tuning succeeded. Model: {job.fine_tuned_model}")
                return job.fine_tuned_model
            elif status in ('failed', 'cancelled', 'expired'):
                print(f"Fine-tuning did not succeed. Status: {status}")
                return None
            time.sleep(poll_interval)
        except Exception as e:
            logging.error(f"Error polling fine-tune job: {e}")
            time.sleep(poll_interval)


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logging.error("OPENAI_API_KEY not found in environment variables.")
        return

    parser = argparse.ArgumentParser(description="Fine-tune an OpenAI model with a JSONL training file.")
    parser.add_argument('--file', type=str, required=True, help='Path to the JSONL training file')
    parser.add_argument('--model', type=str, default='gpt-4.1-2025-04-14', help='Base model to fine-tune')
    parser.add_argument('--poll-interval', type=int, default=30, help='Polling interval in seconds')
    args = parser.parse_args()

    client = OpenAI()

    # Upload training file
    training_file_id = upload_training_file(client, args.file)

    # Create fine-tune job
    job_id = create_fine_tune_job(client, args.model, training_file_id)

    # Poll for job completion
    model_name = poll_fine_tune_job(client, job_id, poll_interval=args.poll_interval)
    if model_name:
        print(f"Fine-tuned model name: {model_name}")
    else:
        print("No fine-tuned model found or job did not succeed.")


if __name__ == "__main__":
    main()
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ARTIFACT_DIR = "artifacts/huggingface"


def bootstrap_finbert():

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    os.environ["HF_HOME"] = ARTIFACT_DIR

    model_id = "ProsusAI/finbert"

    print("Downloading FinBERT...")

    AutoTokenizer.from_pretrained(model_id)
    AutoModelForSequenceClassification.from_pretrained(model_id)

    print("FinBERT bootstrap complete.")


if __name__ == "__main__":
    bootstrap_finbert()

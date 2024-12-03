import os
import sys
from time import sleep

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.LLMClient import LLMClient

PROMPT_MAX_ATTEMPTS = 5
RESPONSE_DELAY = 1


class LLMClassifier:
    def __init__(self, llm_client: LLMClient, file_path: str, output_file: str):
        self.llm_client = llm_client
        self.df = pd.read_csv(file_path)
        self.output_file = output_file

        # Initialise lists for storing predictions and correctness
        self.predictions = []
        self.prediction_correct = []

        # Possible countries (for the prompt generation)
        self.possible_countries = set(self.df["country"])

    def process_data(self):
        """Clean and prepare the data for LLM input."""
        if self.df is None:
            raise ValueError("Data not loaded properly.")

        # Define the header for the output file
        header = ["id", "description", "price", "points", "variety", "actual_country", "predicted_country",
                  "prediction_correct"]

        # Create the output file if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, mode="w", newline='', encoding='utf-8') as f:
                writer = pd.DataFrame(columns=header)
                writer.to_csv(f, index=False)

        # Load already processed IDs from the output file
        processed_ids = set()
        if os.path.exists(self.output_file):
            processed_data = pd.read_csv(self.output_file)
            if "id" in processed_data.columns:
                processed_ids = set(processed_data["id"])

        # Loop through the rows and generate a prompt for each wine
        idx: int

        for idx, row in self.df.iterrows():
            wine_id = row["id"]

            # Skip rows that are already processed
            if wine_id in processed_ids:
                continue

            # Initialise variables for retry logic
            attempts = 0
            predicted_country = None

            while attempts < PROMPT_MAX_ATTEMPTS:
                # Create a concise prompt for the LLM
                prompt = (
                    f"Predict the country of origin for this wine:\n"
                    f"Description: '{row['description']}'\n"
                    f"Price: ${row['price']}, Rating: {row['points']}/100, Variety: '{row['variety']}'.\n"
                    f"Possible countries: [{', '.join(self.possible_countries)}].\n"
                    f"Respond with the country's name as a single word."
                )

                # Get the LLM's response
                response = self.llm_client.get_response(prompt).strip()

                # Validate the response
                if response in self.possible_countries:
                    predicted_country = response
                    break  # Exit the retry loop if the response is valid
                else:
                    print(f"Invalid response: '{response}'. Retrying... ({attempts + 1}/{PROMPT_MAX_ATTEMPTS})")
                    attempts += 1
                    sleep(RESPONSE_DELAY)

            # If no valid response was received after max attempts, skip this row
            if predicted_country is None:
                print(f"Failed to get a valid response for wine ID {wine_id} after {PROMPT_MAX_ATTEMPTS} attempts. "
                      f"Skipping...")
                continue

            # Check if the prediction is correct
            is_correct = 1 if predicted_country == row["country"] else 0

            # Prepare the row to save
            row_to_save = {
                "id": wine_id,
                "description": row["description"],
                "price": row["price"],
                "points": row["points"],
                "variety": row["variety"],
                "actual_country": row["country"],
                "predicted_country": predicted_country,
                "prediction_correct": is_correct,
            }

            # Append the row to the CSV file
            with open(self.output_file, mode="a", newline="", encoding="utf-8") as f:
                writer = pd.DataFrame([row_to_save])
                writer.to_csv(f, header=False, index=False)

            # Print progress
            current_progress = idx + 1
            total_rows = len(self.df)
            print(f"Processed {current_progress}/{total_rows} rows ({(current_progress / total_rows) * 100:.2f}%)")

            # Sleep to respect rate limits for API calls
            sleep(RESPONSE_DELAY)

    def compute_metrics(self):
        """Compute and print evaluation metrics based on predictions."""
        # Read the results CSV
        results = pd.read_csv(self.output_file)

        # Compute overall accuracy
        total_predictions = len(results)
        correct_predictions = results["prediction_correct"].sum()
        accuracy = correct_predictions / total_predictions * 100

        print(f"Overall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correct)")

        # Compute per-country accuracy
        country_metrics = (
            results.groupby("actual_country")["prediction_correct"]
            .agg(accuracy=lambda x: x.sum() / len(x) * 100)
            .reset_index()
        )
        print("\nPer-Country Accuracy:")
        print(country_metrics)

        # Print class distribution (number of wines per country)
        print("\nClass Distribution:")
        class_distribution = results["actual_country"].value_counts()
        print(class_distribution)

        # Print prediction distribution (number of wines predicted per country)
        print("\nPrediction Distribution:")
        prediction_distribution = results["predicted_country"].value_counts()
        print(prediction_distribution)

        # Generate confusion matrix
        print("\nConfusion Matrix:")
        actual = results["actual_country"]
        predicted = results["predicted_country"]
        conf_matrix = pd.DataFrame(
            confusion_matrix(actual, predicted, labels=actual.unique()),
            index=actual.unique(),
            columns=actual.unique(),
        )
        print(conf_matrix)

        # Classification report (precision, recall, f1-score)
        print("\nClassification Report:")
        print(classification_report(actual, predicted))

    def run(self):
        """Run the preprocessing and prediction steps."""
        self.process_data()

        # Compute and display evaluation metrics after all predictions
        self.compute_metrics()

    def clearPreviousEntries(self):
        if os.path.exists(self.output_file):
            print("Found previous entries. Deleting.")
            os.remove(self.output_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python LLMClassifier.py <API_KEY>")
        exit(-1)

    llm = LLMClient(api_key=sys.argv[1])
    classifier = LLMClassifier(llm, file_path="../data/wine_quality_1000.csv",
                               output_file="../output/predictions_1000.csv")
    classifier.process_data()
    classifier.compute_metrics()

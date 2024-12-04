import ast
import os
import re
import sys
from time import sleep

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from LLMClient import LLMClient

PROMPT_MAX_ATTEMPTS = 5
RESPONSE_DELAY = 0.5

WINE_FEATURES = {
    "Fruity Favour": ["berry", "citrus", "apple", "peach", "plum", "tropical fruit"],
    "Spicy Favour": ["pepper", "cinnamon", "nutmeg", "clove"],
    "Earthy Favour": ["mushroom", "forest floor", "truffle", "soil", "leather"],
    "Floral Favour": ["rose", "violet", "jasmine", "honeysuckle"],
    "Oaky Favour": ["oak", "vanilla", "toast", "smoke"],
    "Nutty Favour": ["almond", "hazelnut", "walnut"],
    "Herbal Favour": ["mint", "eucalyptus", "sage", "rosemary"],
    "Mineral Favour": ["flint", "slate", "chalk", "saline"],
    "Sweet Favour": ["honey", "caramel", "molasses"],

    "Body (Weight/Texture)": ["light-bodied", "medium-bodied", "full-bodied", "creamy", "silky", "tannic"],
    "Acidity": ["high acidity", "medium acidity", "low acidity", "crisp", "zesty"],
    "Sweetness Level": ["bone dry", "dry", "off-dry", "sweet", "very sweet"],
    "Alcohol Level (Perception)": ["light", "medium", "high", "hot"],
    "Ageability": ["ready to drink", "needs aging", "long-term aging potential", "over the hill"],
    "Food Pairing / Occasions": [
        "dinner", "casual", "celebration", "aperitif", "barbecue",
        "cheese pairing", "dessert pairing", "seafood pairing"
    ],
    "Tannins": ["low tannins", "medium tannins", "high tannins", "smooth", "velvety", "chewy"],
    "Region/Terroir": ["mineral", "volcanic", "coastal", "mountain"],
    "Color": ["pale", "deep", "ruby", "garnet", "golden", "straw", "amber"],
    "Finish": ["short", "medium", "long", "complex", "crisp", "smooth"],
    "Aging Method": ["stainless steel", "oak barrel", "neutral barrel", "concrete", "amphora", "no oak"],
    "Climate Characteristics": ["cool climate", "warm climate", "hot climate", "dry", "humid"],
    "Intensity/Aroma": ["subtle", "pronounced", "intense", "muted"],
    "Style": ["traditional", "modern", "natural", "organic", "biodynamic"],
    "Unique/Standout Notes": ["smoky", "meaty", "salty", "exotic", "perfumed", "zesty"]
}


class LLMFeatureExtractor:
    def __init__(self, llm_client, file_path: str, output_file: str, external_words_file: str = ""):
        """Initialises the feature extractor with the LLM client and file paths."""
        self.llm_client = llm_client
        self.file_path = file_path
        self.output_file = output_file
        self.external_words_file = external_words_file

        # Load data from the CSV
        self.df = pd.read_csv(self.file_path)

    def feature_extraction(self):
        """Extract features from wine descriptions using the LLM and append results to the output CSV."""
        if self.df is None:
            raise ValueError("Data not loaded properly.")

        # Create the output CSV file if it doesn't exist
        feature_columns = ["id", "country", "description"] + list(WINE_FEATURES.keys())
        if not os.path.exists(self.output_file):
            with open(self.output_file, mode="w", newline='', encoding='utf-8') as f:
                writer = pd.DataFrame(columns=feature_columns)
                writer.to_csv(f, index=False)

        # Load already processed IDs
        processed_ids = set()
        if os.path.exists(self.output_file):
            processed_data = pd.read_csv(self.output_file)
            if "id" in processed_data.columns:
                processed_ids = set(processed_data["id"])

        # Iterate through rows in the DataFrame
        idx: int
        for idx, row in self.df.iterrows():
            wine_id = row["id"]

            # Skip rows already processed
            if wine_id in processed_ids:
                continue

            # Initialise a dictionary to store extracted features for this row
            extracted_features = {"id": wine_id, "country": row["country"], "description": row["description"]}

            # Process each feature in WINE_FEATURES
            for feature, categories in WINE_FEATURES.items():
                attempts = 0
                feature_value = None

                while attempts < PROMPT_MAX_ATTEMPTS:
                    # Construct the prompt
                    prompt = (
                        f"From the following wine description, determine which category the wine belongs to for "
                        "'{feature}':\n"
                        f"Description: '{row['description']}'.\n"
                        f"Possible categories: {categories}.\n"
                        f"If the description does not contain information relevant to '{feature}', respond with "
                        f"'NULL' otherwise response with a single word."
                    )

                    # Get the LLM's response
                    response = self.llm_client.get_response(prompt).strip()

                    # Validate the response
                    if response in categories or response == "NULL":
                        feature_value = response if response != "NULL" else ""
                        break  # Valid response received, exit retry loop
                    else:
                        print(
                            f"Invalid response: '{response}' for feature '{feature}'. Retrying... ({attempts + 1}/"
                            f"{PROMPT_MAX_ATTEMPTS})")
                        attempts += 1
                        sleep(RESPONSE_DELAY)

                # Handle the case where no valid response was received
                if feature_value is None:
                    print(f"Failed to extract '{feature}' for wine ID {wine_id}. Setting it as empty.")
                    feature_value = ""

                # Save the feature value
                extracted_features[feature] = feature_value

            # Append the extracted features to the CSV
            with open(self.output_file, mode="a", newline="", encoding="utf-8") as f:
                writer = pd.DataFrame([extracted_features])
                writer.to_csv(f, header=False, index=False)

            # Print progress
            current_progress = idx + 1
            total_rows = len(self.df)
            print(f"Processed {current_progress}/{total_rows} rows ({(current_progress / total_rows) * 100:.2f}%)")

            # Sleep to respect API rate limits
            sleep(RESPONSE_DELAY)

    def text_vectorisation(self):
        """Performs LLM-based feature extraction and vectorises text descriptions."""
        if self.df is None:
            raise ValueError("Data not loaded properly.")

        # Define header for the output file
        header = ["id", "extracted_words"]

        # Create the output file if it doesn't exist
        if not os.path.exists(self.external_words_file):
            with open(self.external_words_file, mode="w", newline='', encoding='utf-8') as f:
                writer = pd.DataFrame(columns=header)
                writer.to_csv(f, index=False)

        # Load already processed IDs from the external file
        processed_ids = set()
        if os.path.exists(self.external_words_file):
            processed_data = pd.read_csv(self.external_words_file)
            if "id" in processed_data.columns:
                processed_ids = set(processed_data["id"])

        print("Performing LLM-based feature extraction...")

        # Loop through the rows and generate a prompt for each wine
        index: int
        for index, row in self.df.iterrows():
            wine_id = row["id"]
            description = row["description"]

            # Skip rows that are already processed
            if wine_id in processed_ids:
                continue

            # Prepare the prompt for LLM
            prompt = (
                "Extract important words or named entities from the wine description below. "
                "Focus on key details like wine types, flavors, regions, or any notable attributes.\n"
                "Description: {description}\n"
                "Respond with a comma-separated list of these key features."
            ).format(description=description)

            # Get response from the LLM
            response = self.llm_client.get_response(prompt).strip()

            # Parse the LLM response to extract relevant words/names
            extracted_words = response.lower().replace(",", "").split()

            # Store the extracted words with the wine ID
            row_to_save = {
                "id": wine_id,
                "extracted_words": extracted_words
            }

            # Append the row to the CSV file (using mode="a" to append)
            with open(self.external_words_file, mode="a", newline='', encoding='utf-8') as f:
                writer = pd.DataFrame([row_to_save])
                writer.to_csv(f, header=False, index=False)

            current_progress = index + 1
            total_rows = len(self.df)
            print(f"Processed {current_progress}/{total_rows} rows ({(current_progress / total_rows) * 100:.2f}%)")

            # Sleep to avoid overwhelming the API with too many requests
            sleep(1)

        print("Finished extracting features and saving to CSV.")

        # Apply TF-IDF
        self._apply_tfidf(pd.read_csv(self.external_words_file))

    def _apply_tfidf(self, extracted_words_df):
        """Vectorise the extracted words using TF-IDF."""
        desc_words = set()

        # Iterate over each row in the DataFrame
        for words in extracted_words_df["extracted_words"]:
            # Split the comma-separated words and add them to the set
            desc_words.update(ast.literal_eval(words))

        # Use TF-IDF Vectoriser to vectorise the words
        tfidf_vectoriser = TfidfVectorizer(tokenizer=lambda t: re.findall(r'\b\w+(?:-\w+)*\b', t.lower()),
                                           token_pattern=None, vocabulary=list(desc_words))
        X = tfidf_vectoriser.fit_transform(extracted_words_df["extracted_words"])

        # Create a DataFrame from the TF-IDF output
        tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf_vectoriser.get_feature_names_out())

        # Concatenate the original data with the TF-IDF features
        final_df = pd.concat([self.df, tfidf_df], axis=1)

        # Save the final dataset to the output file
        final_df.to_csv(self.output_file, index=False)
        print(f"Final dataset with TF-IDF features saved to {self.output_file}.")

    def clearPreviousEntries(self):
        if os.path.exists(self.external_words_file):
            print("Found previous entries. Deleting.")
            os.remove(self.external_words_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python LLMClassifier.py <API_KEY>")
        exit(-1)

    llm = LLMClient(api_key=sys.argv[1])
    lfe = LLMFeatureExtractor(llm, file_path="../data/wine_quality_1000.csv",
                              output_file="../output/wine_quality_more_features_1000.csv")
    lfe.feature_extraction()

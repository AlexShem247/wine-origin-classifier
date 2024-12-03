import re
import sys
from time import sleep

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.LLMClient import LLMClient, LLMClientError

PROMPT_MAX_ATTEMPTS = 5
RESPONSE_DELAY = 1

def custom_tokenizer(text):
    # This regex keeps hyphenated words together
    return re.findall(r'\b\w+(?:-\w+)*\b', text.lower())


class LLMFeatureExtractor:
    def __init__(self, llm_client: LLMClient, file_path: str, output_file: str):
        self.llm_client = llm_client
        self.output_file = output_file
        self.df = pd.read_csv(file_path)

    def text_vectorisation(self, max_retries=5, prompt_max_attempts=5, response_delay=1):
        """Performs LLM-based feature extraction and vectorizes text descriptions."""
        if self.df is None:
            raise ValueError("Data not loaded properly.")

        print("Performing LLM-based feature extraction...")

        # Define the prompt template for LLM
        prompt_template = (
            "Extract important words or named entities from the wine description below. "
            "Focus on key details like wine types, flavors, regions, or any notable attributes.\n"
            "Description: {description}\n"
            "Respond with a comma-separated list of these key features."
        )

        # Initialize a list to store sets of extracted words/features for each row
        all_extracted_words = []

        for index, row in self.df.iterrows():
            description = row["description"]
            retries = 0

            while retries < prompt_max_attempts:
                try:
                    # Format the prompt with the description
                    prompt = prompt_template.format(description=description)

                    # Get response from the LLM
                    response = self.llm_client.get_response(prompt).strip()

                    # Parse the LLM response to extract relevant words/names
                    extracted_words = set(response.lower().replace(",", "").split())

                    # Append the set of extracted words to the list
                    all_extracted_words.append(extracted_words)
                    break  # Exit retry loop if successful

                except LLMClientError as e:
                    print(f"Error: {e}. Retrying ({retries + 1}/{prompt_max_attempts})...")
                    retries += 1
                    sleep(response_delay)
            else:
                print(f"Failed to process row {index} after {prompt_max_attempts} retries.")

            sleep(response_delay)

        # Apply TF-IDF vectorization to the extracted words
        self._apply_tfidf(all_extracted_words)

        # Output the file with the new features
        self.df.to_csv(self.output_file, index=False)
        print(f"Enhanced dataset saved to {self.output_file}.")

    def _apply_tfidf(self, all_extracted_words):
        """Applies TF-IDF vectorization to the extracted words."""
        # Create a list of unique words from all rows
        all_unique_words = sorted(set(word for words in all_extracted_words for word in words))

        # Convert the sets of extracted words into a string format for TF-IDF
        word_lists = [" ".join(words) for words in all_extracted_words]

        # Initialize the TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None, vocabulary=all_unique_words)

        # Apply TF-IDF vectorization to the descriptions
        tfidf_matrix = tfidf_vectorizer.fit_transform(word_lists)

        # Convert the sparse matrix to a DataFrame and set the column names
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # Concatenate the new TF-IDF features with the original dataset
        self.df = pd.concat([self.df, tfidf_df], axis=1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python LLMClassifier.py <API_KEY>")
        exit(-1)

    llm = LLMClient(api_key=sys.argv[1])
    lfe = LLMFeatureExtractor(llm, file_path="../data/wine_quality_10.csv",
                              output_file="../output/wine_quality_features+.csv")
    lfe.text_vectorisation()

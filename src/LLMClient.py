from time import sleep

import requests


class LLMClientError(Exception):
    """Custom exception for handling LLM Client API errors."""
    pass


class LLMClient:
    MAX_RETRIES = 5

    def __init__(self, api_key, url="https://candidate-llm.extraction.artificialos.com/v1/chat/completions"):
        self.api_key = api_key
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

    def get_response(self, prompt, model="gpt-4o-mini", temperature=0.7):
        """Send a prompt to the LLM and get a response with retry logic for rate limits."""
        retry_count = 0  # Counter for retries

        while retry_count < self.MAX_RETRIES:
            # Define the request payload
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }

            # Make the POST request
            response = requests.post(self.url, headers=self.headers, json=data)

            # Check if the response was successful
            if response.status_code == 200:
                response_json = response.json()

                # Validate response structure
                choices = response_json.get("choices")
                if not choices or "message" not in choices[0]:
                    raise LLMClientError("Invalid response format: Missing 'choices' or 'message'.")

                # Extract and return the assistant's message
                assistant_message = choices[0]["message"].get("content", "No response")
                return assistant_message

            # Handle rate limit error (429 Too Many Requests)
            elif response.status_code == 429:
                retry_count += 1
                sleep(0.5)

                # Handle other errors
            else:
                # raise LLMClientError(f"Error {response.status_code}: {response.text}")
                sleep(5)
                return self.get_response(prompt)

        # If all retries fail, raise an exception
        sleep(5)
        return self.get_response(prompt)

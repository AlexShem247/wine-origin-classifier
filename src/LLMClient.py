import requests


class LLMClientError(Exception):
    """Custom exception for handling LLM Client API errors."""
    pass


class LLMClient:
    def __init__(self, api_key, url="https://candidate-llm.extraction.artificialos.com/v1/chat/completions"):
        self.api_key = api_key
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

    def get_response(self, prompt, model="gpt-4o-mini", temperature=0.7):
        # Define the request payload
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }

        # Make the POST request
        response = requests.post(self.url, headers=self.headers, json=data)

        # Check if the response was successful
        if response.status_code != 200:
            raise LLMClientError(f"Error {response.status_code}: {response.text}")

        response_json = response.json()

        # Check if the response contains the expected fields
        choices = response_json.get("choices")
        if not choices or "message" not in choices[0]:
            raise LLMClientError("Invalid response format: Missing 'choices' or 'message'.")

        # Extract the assistant's message
        assistant_message = choices[0]["message"].get("content", "No response")
        return assistant_message

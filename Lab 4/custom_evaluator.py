from typing import Any, Dict, List, Optional
import re
import time
import requests
import tqdm
import json


class CustomEvaluator():
    """
    Evaluator that uses RAGAS metrics to score RAG-based QA performance.
    """

    def __init__(self, evaluator_llm_info):
        self.evaluator_llm = evaluator_llm_info.get('eval_retrieval_model', '')
        self.temperature = evaluator_llm_info.get('temp_retrieval_llm', 0.1)
        self.pvalue = evaluator_llm_info.get('pvalue_retrieval_llm', 0.1)
        prompt_file_path = '../data/eval_prompt.json'
        with open(prompt_file_path, 'r') as f:
            self.prompt = json.load(f)

    def _load_system_prompt(self, item) -> str:
        """
        Load system prompt from file. Falls back to default if file not found.

        Returns:
            System prompt string
        """
        # Get the project root directory (where the package is installed)
        # project_root = Path(__file__).parent
        prompt = self.prompt['eval_prompt']
        question = item.get('question', '')
        ground_truth = item.get('expected_answer', '')
        prediction = item.get('generated_answer', '')
        eval_prompt = prompt.format(user_query=question, ground_truth=ground_truth, prediction=prediction)
        return eval_prompt

    def create_payload(self, prompt: str) -> Dict[str, Any]:
        """
        Create payload with system prompt and user message.

        Args:
            prompt: Formatted prompt string

        Returns:
            Payload dictionary for API request
        """
        payload = {
            "model": self.evaluator_llm,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature, "top_p": 0.1
        }

        return payload

    # def create_prompt(self, item) -> str:
    #     """
    #     Create prompt with context and question.
    #
    #     Args:
    #         item: Dictionary containing query and search results
    #
    #     Returns:
    #         Formatted prompt string
    #     """
    #     question = item.get('question', '')
    #     ground_truth = item.get('expected_answer', '')
    #     prediction = item.get('generated_answer', '')
    #     # print(f"Question: {question}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n")
    #     return f"Question: {question}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"

    def parse_response(self, response: str) -> Dict:
        """
        Return a tuple of (explanation, score) from the response,
        where score is 0 if the prediction is wrong, 1 if the prediction is correct.

        Args:
            response: Response string containing JSON-formatted evaluation result

        Returns:
            Tuple containing (explanation, score)
            - explanation: String explaining the evaluation result
            - score: 1 if correct, 0 if incorrect, -1 if parsing error

        Examples:
            Corner case 1:
                {"explanation": ...}
                Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
                {...}

            Corner case 2:
                {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        """
        matches = re.findall(r"{([^}]*)}", response)
        text = ""
        for match in matches:
            text = "{" + match + "}"
        try:
            score = -1
            # Pattern to match the score
            score_pattern = r'"score"\s*:\s*(\d+)'
            score_match = re.search(score_pattern, text)
            if score_match:
                score = int(score_match.group(1))
                if score != 0 and score != 1:
                    raise Exception("bad score: " + response)
            else:
                return {"message": response, "score": -1}

            # Pattern to match the explanation
            explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
            explanation_match = re.search(explanation_pattern, text)
            if explanation_match:
                explanation = explanation_match.group(1)
                return {"message": explanation, "score": score}
            else:
                return {"message": text, "score": score}
        except Exception as e:
            print(f"Parsing Error with resp: {response}")
            print(f"Error: {e}")
            return {"message": response, "score": -1}

    def _make_api_request(self, payload: Dict) -> Dict:
        """
        Make API request with retry logic.

        Args:
            payload (Dict): Request payload.

        Returns:
            Dict: API response data.

        Raises:
            AuthenticationError: If API key is invalid.
            RequestError: For other HTTP request issues.
        """
        global response
        api_key = 'flt_3cqiawdagpy75ae9yhah1nh2o2us53mco0h10e0ni58d49do'
        if not api_key:
            raise ValueError("API_KEY is not set in the environment or config")

        headers = {"Authorization": api_key, "Content-Type": "application/json"}
        max_retries = 3
        base_wait_time = 1  # Initial wait time in seconds

        for attempt in range(max_retries):
            try:
                response = requests.request("POST",
                                            'https://fphcciizk3.us-east-1.awsapprunner.com/api/v1/chat/completions',
                                            headers=headers, json=payload)
                response_time = response.elapsed.total_seconds()

                if response.status_code == 429:  # Handle rate limiting
                    retry_after = int(response.headers.get('Retry-After', 60))
                    # self.logger.warning(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                response_json = response.json()
                response_json['response_time'] = response_time
                return response_json


            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    raise

                wait_time = base_wait_time * (2 ** attempt)  # Exponential backoff
                if attempt < max_retries - 1:  # Don't log on last attempt
                    time.sleep(wait_time)
                else:
                    raise

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise

        return response.json()  # In case we get here after rate limit retries

    def evaluate(
            self,
            data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # example to fetch metrics, use like this
        for each in tqdm.tqdm(data, desc="Processing"):
            prompt = self._load_system_prompt(each)
            payload = self.create_payload(prompt)
            response = self._make_api_request(payload)
            each['response'] = self.parse_response(response['choices'][0]['message']['content'])
        return data

    def evaluate_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Evaluate the results of the processed data.

        Args:
            results (List[Dict[str, Any]]): List of evaluation results.
        """
        final_results = {}
        for _id, result in results.items():
            full_score_count = sum(1 for item in result if item['response']['score'] == 1)
            final_results[_id] = {"number of samples correct": full_score_count}
        return final_results
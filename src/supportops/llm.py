import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = 'qwen2.5:1.5b'
DEFAULT_BASE_URL = 'http://127.0.0.1:11434'
DEFAULT_TIMEOUT = 60

def generate_response(prompt : str, model : str | None = None, temperature : float = 0.2, timeout : int = DEFAULT_TIMEOUT) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Prompt must be a non-empty string.')
    if not 0 <= temperature <= 2:
        raise ValueError('Temperature must be between 0 and 2.')
    if timeout <= 0:
        raise ValueError('Timeout must be greater than zero.')
    selected_model = model or os.getenv('SUPPORTOPS_LLM_MODEL', DEFAULT_MODEL)
    base_url = os.getenv('OLLAMA_BASE_URL', DEFAULT_BASE_URL).rstrip('/')
    payload = {
        'model' : selected_model,
        'prompt' : prompt.strip(),
        'stream' : False,
        'options' : {
            'temperature' : temperature,
            'num_predict' : 180
        }
    }
    request = Request(
        url = f'{base_url}/api/generate',
        data = json.dumps(payload).encode('utf-8'),
        headers = {'Content-Type' : 'application/json'},
        method = 'POST'
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. "
            "Check that the Ollama server is running."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama did not respond within {timeout} seconds.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned an invalid JSON response.") from exc
    generated_text = result.get("response")
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise RuntimeError("Ollama returned no usable generated text.")
    return generated_text.strip()
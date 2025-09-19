import requests

def get_ollama_response(user_prompt, model="gemma:2b", temperature=0.9):
    """
    Sends a prompt to the Ollama API and returns the model's response.
    Now accepts model name and temperature as parameters.
    """
    ollama_url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,  # Now uses the model parameter
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": temperature  # Now uses the temperature parameter
        }
    }

    try:
        print(f"Sending request to Ollama using {model}, temp {temperature}: {user_prompt[:50]}...")
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Received response from Ollama.")
        return result["response"]
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is it running?"
    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama: {e}"
    except KeyError:
        return "Error: Unexpected response format from Ollama."
# app/handlers/text_handler.py

import requests

def process_text(text: str) -> dict:
    """
    Envoie le texte à l'API LM Studio locale pour analyse de sentiment.
    """
    try:
        url = "http://localhost:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "lmstudio",
            "messages": [
                {"role": "system", "content": "You are an assistant specialized in sentiment analysis."},
                {"role": "user", "content": text}
            ],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return {
            "message": text,
            "sentiment_analysis": content.strip()
        }
    except requests.RequestException as e:
        return {"error": str(e)}

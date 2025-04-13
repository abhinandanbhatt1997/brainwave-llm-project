import openai
from config.settings import OPENAI_API_KEY

def query_llm(state: int):
    """Get GPT-4 response based on brain state"""
    openai.api_key = OPENAI_API_KEY
    prompt = (
        "The user is focused. Suggest a productivity hack:" if state == 1 
        else "The user is relaxed. Suggest a mindfulness activity:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']
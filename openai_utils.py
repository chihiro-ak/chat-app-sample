import os

import openai

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))


def response_openai(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"エラーが発生しました: {e}"


def text_to_vector(text_data):
    try:
        response = client.embeddings.create(
            input=text_data, model="text-embedding-ada-002", dimensions=None
        )
        return response.data[0].embedding
    except Exception as e:
        return f"エラーが発生しました: {e}"

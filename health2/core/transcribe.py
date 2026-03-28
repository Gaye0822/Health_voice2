import openai
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(file_path: str) -> str:
    """
    Converts audio file to raw text using OpenAI Whisper.
    Returns raw transcript - no normalization here.
    """
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt=(
                "Personal health tracking voice note including supplements, "
                "biomarkers, devices, symptoms and measurements."
            )
        )
    return response.text
import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)
    
from gtts import gTTS
import io

def convert_text_to_speech(text_to_speak):
    """
    Converts a given text string into spoken audio data (MP3 format).

    This function uses the gTTS (Google Text-to-Speech) library to generate
    audio from text. To avoid writing temporary files to disk, it saves the
    resulting MP3 data directly into an in-memory binary buffer (io.BytesIO).

    Args:
        text_to_speak (str): The text content that needs to be converted
                             into speech.

    Returns:
        bytes: The raw byte data of the generated MP3 audio, ready to be
               played by an audio player component. Returns None if the
               input text is empty or an error occurs.
    """
    if not text_to_speak:
        return None
        
    try:
        tts = gTTS(text_to_speak, lang='en')
        mp3_data_buffer = io.BytesIO()
        tts.write_to_fp(mp3_data_buffer)
        mp3_data_buffer.seek(0)
        return mp3_data_buffer.getvalue()
    except Exception as e:
        print(f"An error occurred during text-to-speech conversion: {e}")
        return None

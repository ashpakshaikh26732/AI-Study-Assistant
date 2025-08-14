import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)
    
from transformers import pipeline

def load_whisper_model(config):
    """
    Loads and initializes the Whisper ASR (Automatic Speech Recognition) pipeline.

    This function creates a transformers pipeline for speech recognition using
    the Whisper model specified in the configuration. It's designed to be
    called once and have its result cached or reused to avoid reloading the
    model repeatedly.

    Args:
        config (dict): The project's configuration dictionary, which must
                     contain the Hugging Face model name for Whisper under
                     the 'voice.whisper_model' key.

    Returns:
        transformers.pipelines.base.Pipeline: The initialized ASR pipeline
        object, ready to be used for transcription.
    """
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=config['voice']['whisper_model'],
        chunk_length_s=30,
        device=0  
    )
    return asr_pipeline

def transcribe_audio(audio_bytes, asr_pipeline):
    """
    Transcribes a given audio input into text using the loaded Whisper pipeline.

    Args:
        audio_bytes (bytes): The raw audio data captured from a source like a
                             microphone.
        asr_pipeline (transformers.pipelines.base.Pipeline): The pre-loaded
                      ASR pipeline object from the `load_whisper_model` function.

    Returns:
        str: The transcribed text from the audio. Returns an empty string
             if transcription fails or produces no text.
    """
    
    result = asr_pipeline(audio_bytes)
    transcribed_text = result.get("text", "").strip()
    return transcribed_text

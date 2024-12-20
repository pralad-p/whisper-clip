import whisper
import torch


class WhisperClient:
    def __init__(self, model_name="medium.en"):
        device = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name,device=self.device)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]

import whisper
import torch
import functools


class WhisperClient:
    def __init__(self, model_name="medium.en"):
        device = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
         # Temporarily override torch.load to add weights_only=True
        original_torch_load = torch.load
        torch.load = functools.partial(torch.load, weights_only=True)
        try:
            # Call load_model
            self.model = whisper.load_model(model_name, device=self.device)
        finally:
            # Restore the original torch.load to avoid affecting other parts of the program
            torch.load = original_torch_load

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]

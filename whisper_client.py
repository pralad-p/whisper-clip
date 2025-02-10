import torch
import whisper
import torch
import functools


class WhisperClient:
    def __init__(self, model_name="medium.en"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        if self.model is None:
            device = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
         # Temporarily override torch.load to add weights_only=True
        original_torch_load = torch.load
        torch.load = functools.partial(torch.load, weights_only=True)
        try:
            # Call load_model
            self.model = whisper.load_model(self.model_name, device=self.device)
        finally:
            # Restore the original torch.load to avoid affecting other parts of the program
            torch.load = original_torch_load

    def unload_model(self):
        if self.model is not None:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete model and clear from memory
            del self.model
            self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio_path):
        if self.model is None:
            self.load_model()
        result = self.model.transcribe(audio_path)
        return result["text"]

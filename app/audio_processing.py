
import torch
import os
from hallo.datasets.audio_processor import AudioProcessor

def process_audio_emb(audio_emb):
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

class AudioProcessing:
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, driving_audio_path, save_path):
        with AudioProcessor(
            self.config.data.driving_audio.sample_rate,
            self.config.data.export_video.fps,
            self.config.wav2vec.model_path,
            self.config.wav2vec.features == "last",
            os.path.dirname(self.config.audio_separator.model_path),
            os.path.basename(self.config.audio_separator.model_path),
            os.path.join(save_path, "audio_preprocess")
        ) as audio_processor:
            audio_emb = audio_processor.preprocess(driving_audio_path)
        return audio_emb
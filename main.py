import base64
import io
import modal
import numpy as np
import torch.nn as nn
import torchaudio.transforms as T
import torch
from pydantic import BaseModel
import soundfile as sf
import librosa
from model import AudioCNN
import requests


app = modal.App("sonimir")


image = (modal.Image.debian_slim().pip_install_from_requirements(
    "requirements.txt").apt_install(["libsndfile1"]).add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(sample_rate=44100, n_fft=1024,  # this has to be kept to 22050 to match the training data, BUT it really should be 44100
                             hop_length=512, n_mels=128, f_min=0, f_max=11025),
            T.AmplitudeToDB(),
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)

        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


class InferenceRequest(BaseModel):
    audio_data: str


# CUDA available for A10G GPU
@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading models on enter...")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load("/models/best_model.pth",
                                map_location=self.device)

        self.classes = checkpoint['classes']

        self.model = AudioCNN(num_classes=len(self.classes))
        # load the best model weights and biases from our training
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print("Model loaded on enter!")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        # production env: frontend -> upload file to s3 -> inference endpoint -> download from s3 bucket
        # sandbox env: frontend -> send file directly -> inference endpoint
        audio_bytes = base64.b64decode(request.audio_data)

        audio_data, sample_rate = sf.read(
            io.BytesIO(audio_bytes), dtype='float32')

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 44100:
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=44100)

        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        with torch.no_grad():
            output, feature_maps = self.model(
                spectrogram, return_feature_maps=True)

            output = torch.nan_to_num(output)
            # dim=0 batch, dim=1 class (batch_size, num_classes)
            probabilities = torch.softmax(output, dim=1)

            top3_probs, top3_indices = torch.topk(probabilities[0], k=3)

            # dog: 0.8, chirping_birds: 0.1,
            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                           for prob, idx in zip(top3_probs, top3_indices)]

            viz_data = {}
            for name, tensor in feature_maps.items():
                # feature maps have the following structure [batch_size, channels, height, width]
                if tensor.dim() == 4:
                    # avg out channels (outputs of the convolutional layer)
                    aggregated_tensor = torch.mean(tensor, dim=1)
                    squeezed_tensor = aggregated_tensor.squeeze(
                        0)  # remove batch size

                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }

                # [batch, channel, height, width] -> [height, width]
                spectrogram_np = spectrogram.squeeze(
                    0).squeeze(0).cpu().numpy()
                clean_spectrogram = np.nan_to_num(spectrogram_np)

                max_samples = 8000
                if len(audio_data) > max_samples:
                    sample_step = len(audio_data) // max_samples
                    waveform_data = audio_data[::sample_step]
                else:
                    waveform_data = audio_data

        response = {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": 44100,
                "duration": len(audio_data) / 44100
            }
        }

        return response


@app.local_entrypoint()
def main():
    audio_data, sample_rate = sf.read("thunder.wav")

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")

    audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    payload = {"audio_data": audio_b64}

    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()

    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get('duration', 0)} seconds")

    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f"  -{pred["class"]} {pred["confidence"]:0.2%}")

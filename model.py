import whisper 
import matplotlib.pyplot as plt 
import librosa 
import numpy as np 
from typing import Dict, Union

## ignore UserWarning
import warnings
warnings.filterwarnings('ignore')


class AudioTranscriber: 
    def __init__(self, model_name: str, audio_file_path: str):
        self.model = self.load_model(model_name)
        self.audio_file = audio_file_path
        self.loaded_audio = self.load_audio(audio_file_path)
        self.whisper_mel_spec = None

    @staticmethod
    def load_model(model_name: str) -> whisper.Whisper:
        """Loads the model"""
        return whisper.load_model(model_name)
    @staticmethod
    def load_audio(audio_file: str) -> np.ndarray:
        """Loads the audio file into the whisper model"""
        return whisper.load_audio(audio_file)


    def plot_melspec(self) -> None: 
        """Plots the log-mel-spectrogram"""
        ## load the audio file
        y, sr = librosa.load(self.audio_file)

        ## extract the mel spectrogram
        mel_spec= librosa.feature.melspectrogram(y=y, sr=sr)

        ## plot the mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()

    ## the output of the following function is a dictionary with a dictionary[str, float], np.ndarray and a string 
    def mel_spec_audio_detection(self) -> Dict[str, Union[Dict[str, float], np.ndarray, str]]:
        """Takes in the loaded audio file and returns the language and the mel spectrogram"""
        ## padding/trimming for the decoder 
        padded_trimmed_audio = whisper.pad_or_trim(self.loaded_audio)
        
        ## extract the log-mel-spec 
        mel_spec = whisper.log_mel_spectrogram(padded_trimmed_audio)
        self.whisper_mel_spec = mel_spec
        ## detect the language
        _, probs = self.model.detect_language(mel_spec)
        ## detected language 
        det_lang = max(probs, key=probs.get)

        return {"language": det_lang, "mel_spec": mel_spec, "probs": probs}

    def transcribe_audio(self, options: whisper.DecodingOptions) -> str:
        """Transcribes the audio file"""
        ## decode the audio
        return whisper.decode(self.model, self.whisper_mel_spec, options).__dict__

    def transcribe_audio_no_trim(self) -> dict:
        """Transcribes the audio without padding or trimming"""
        return self.model.transcribe(self.audio_file)

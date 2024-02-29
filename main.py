import whisper
import pyaudio
import numpy as np
from tqdm import tqdm
from debug.instrumentor import *


class WisperLoop:
    def __init__(self) -> None:
        self.init_speech_recognization_model()
        self.is_listening = True
        pass

    @PROFILE_FUNCTION
    def init_speech_recognization_model(self):
        self.model = whisper.load_model("tiny", download_root="./model")
        # self.model = whisper.load_model("tiny.en", download_root="./model")
        # self.model = whisper.load_model("base", download_root="./model")
        # self.model = whisper.load_model("base.en", download_root="./model")
        # self.model = whisper.load_model("small", download_root="./model")
        # self.model = whisper.load_model("small.en", download_root="./model")
        # self.model = whisper.load_model("medium", download_root="./model")
        # self.model = whisper.load_model("medium.en", download_root="./model")
        # self.model = whisper.load_model("large", download_root="./model")
        # self.model = whisper.load_model("large-v1", download_root="./model")
        # self.model = whisper.load_model("large-v2", download_root="./model")
        print("Model loading completed.")

    @PROFILE_FUNCTION
    def start_recorder(self):
        self.channel = 1
        self.frame_rate = 16000
        self.record_seconds = 4
        self.audio_format = pyaudio.paInt16
        self.chunk = 1024
        self.p = pyaudio.PyAudio()
        self.recorder = self.p.open(
            format=self.audio_format,
            channels=self.channel,
            rate=self.frame_rate,
            input=True,
            input_device_index=1,
            frames_per_buffer=self.chunk,
        )
        print("Audio record start.")

    @PROFILE_FUNCTION
    def close_recorder(self):
        self.recorder.stop_stream()
        self.recorder.close()
        self.p.terminate()
        print("Audio record closed.")

    def run(self):
        self.start_recorder()
        frames = []
        progress_bar = tqdm(
            total=int(self.frame_rate * self.record_seconds / self.chunk),
            desc=f"Speech recognization Once every {self.record_seconds} seconds",
            unit="frame",
        )
        while self.is_listening:
            with PROFILE_SCOPE("Speech data collection"):
                frame = self.recorder.read(self.chunk)
                frame = np.frombuffer(frame, np.int16).flatten().astype(np.float32)
                frames.append(frame)
                progress_bar.update(1)
            if len(frames) >= int(self.frame_rate * self.record_seconds / self.chunk):
                with PROFILE_SCOPE("Speech data process"):
                    audio = np.concatenate(frames, axis=0) / 32768.0
                    # print(audio.shape)
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
                    # _, probs = model.detect_language(mel)
                    # print(mel.shape)
                    # print(f"Detected language: {max(probs, key=probs.get)}")
                with PROFILE_SCOPE("Speech recognizaion"):
                    options = whisper.DecodingOptions(fp16=False)
                    result = whisper.decode(self.model, mel, options)
                    # print the recognized text
                    print(f"\n{result.text}")
                frames = []
                progress_bar.reset()
        self.close_recorder()


if __name__ == "__main__":
    PROFILE_BEGIN_SESSION("project_init", "project_init_profile.json")
    app = WisperLoop()
    PROFILE_END_SESSION()
    PROFILE_BEGIN_SESSION("project_running", "project_running_profile.json")
    app.run()
    PROFILE_END_SESSION()

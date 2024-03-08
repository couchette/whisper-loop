import whisper
import pyaudio
import numpy as np
from tqdm import tqdm
from debug.instrumentor import *
import matplotlib.pyplot as plt
import wave
import math
import webrtcvad


class WhisperModelCard:
    def __init__(self):
        self.__sliding_dur = 30
        self.__model = whisper.load_model("tiny", download_root="./model")
        # self.__model = whisper.load_model("tiny.en", download_root="./model")
        # self.__model = whisper.load_model("base", download_root="./model")
        # self.__model = whisper.load_model("base.en", download_root="./model")
        # self.__model = whisper.load_model("small", download_root="./model")
        # self.__model = whisper.load_model("small.en", download_root="./model")
        # self.__model = whisper.load_model("medium", download_root="./model")
        # self.__model = whisper.load_model("medium.en", download_root="./model")
        # self.__model = whisper.load_model("large", download_root="./model")
        # self.__model = whisper.load_model("large-v1", download_root="./model")
        # self.__model = whisper.load_model("large-v2", download_root="./model")

    @property
    def model(self):
        return self.__model

    @property
    def sliding_dur(self):
        return self.__sliding_dur


class WisperLoop:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.is_listening = True
        self.cache_speech = []
        self.cache_frames = []
        self.speech = ""
        self.time_speed_last_recognization = 0.01
        self.init_vad()
        self.init_recorder()
        self.init_speech_recognization_model()
        pass

    def __del__(self):
        if self.recorder:
            self.close_recorder()

    def init_vad(self):
        self.__vad = webrtcvad.Vad()
        self.__vad.set_mode(3)

    @PROFILE_FUNCTION
    def init_speech_recognization_model(self):
        whisper_model_card = WhisperModelCard()
        self.model = whisper_model_card.model
        self.cache_dur = 3  # whisper_model_card.sliding_dur
        self.cache_chunk_num = math.ceil(self.cache_dur / self.time_speed_each_chunk)
        print("Model loading completed.")

    def init_recorder(self):
        self.channel = 1
        self.frame_rate = 16000
        self.audio_format = pyaudio.paInt16
        self.chunk = 480
        self.time_speed_each_chunk = self.chunk / self.frame_rate

    @PROFILE_FUNCTION
    def start_recorder(self):
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

    def wave_read(self):
        while True:
            yield self.recorder.read(self.chunk)

    def save_wave(self, frames, wave_output_filename="output.wav"):
        wf = wave.open(wave_output_filename, "wb")
        wf.setnchannels(self.channel)
        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
        wf.setframerate(self.frame_rate)
        wf.writeframes(b"".join(frames))
        wf.close()

    def run(self):
        self.start_recorder()
        chunks_count = 0
        while True:
            chunks_num = math.ceil(
                self.time_speed_last_recognization / self.time_speed_each_chunk
            )
            for index, wave_data in enumerate(self.wave_read()):
                print(len(wave_data))
                is_speech = self.__vad.is_speech(wave_data, sample_rate=self.frame_rate)
                if is_speech:
                    print("有声音活动")
                else:
                    print("没有声音活动")
                self.cache_frames.append(wave_data)
                chunks_count += 1
                if index >= chunks_num:
                    break
            # speech recogization
            start_time_speech_recognization = time.time()
            cache_frames_np = [
                np.frombuffer(x, np.int16).flatten().astype(np.float32)
                for x in self.cache_frames
            ]

            audio = np.concatenate(cache_frames_np, axis=0) / 32768.0
            # plt.plot(np.linspace(0, len(audio), len(audio)), audio)
            # plt.pause(0.01)
            audio = whisper.pad_or_trim(audio)
            # plt.plot(np.linspace(0, len(audio), len(audio)), audio)
            # plt.pause(0.01)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(self.model, mel, options)
            self.time_speed_last_recognization = (
                time.time() - start_time_speech_recognization
            )
            self.cache_speech.append(result.text)
            print(f"\n recog: {self.cache_speech[-1]}\n")

            if chunks_count >= self.cache_chunk_num:
                chunks_count = 0
                self.speech += self.cache_speech[-1]
                self.save_wave(frames=self.cache_frames)
                with open("speech.txt", "w", encoding="utf-8") as f:
                    f.write(self.speech)
                self.cache_frames = []
                self.cache_speech = []
                break

        self.close_recorder()


if __name__ == "__main__":
    PROFILE_BEGIN_SESSION("project_init", "project_init_profile.json")
    app = WisperLoop()
    PROFILE_END_SESSION()
    PROFILE_BEGIN_SESSION("project_running", "project_running_profile.json")
    app.run()
    PROFILE_END_SESSION()

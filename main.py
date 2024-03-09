import whisper
import pyaudio
import numpy as np
from debug.instrumentor import *
import matplotlib.pyplot as plt
import wave
import math
import webrtcvad
import queue
from core.model_card import WhisperModelCard
from core.text_compiler import TextCompiler


class WisperLoop:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.is_listening = False
        self.threads = []
        self.cache_speech = []
        self.wait_process_frames_queue = queue.Queue()
        self.cache_frames = []
        self.cache_frames_status = []
        self.pause_chunks_num_threshold = 5
        self.speech_content = ""
        self.time_speed_last_recognization = 0.01
        self.chunks_count = 0
        self.init_vad()
        self.init_text_compiler()
        self.init_recorder()
        self.init_speech_recognization_model()
        pass

    def __del__(self):
        self.is_listening = False
        for t in self.threads:
            t.join()
        if self.recorder:
            self.close_recorder()

    def init_vad(self):
        self.__vad = webrtcvad.Vad()
        self.__vad.set_mode(1)

    def init_text_compiler(self):
        self.__text_compiler = TextCompiler()
        self.pause_flag = self.__text_compiler.pause_flag

    @PROFILE_FUNCTION
    def init_speech_recognization_model(self):
        whisper_model_card = WhisperModelCard()
        self.model = whisper_model_card.model
        self.cache_dur = whisper_model_card.sliding_dur
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

    def save_wave(self, frames, wave_output_filename="output.wav"):
        wf = wave.open(wave_output_filename, "wb")
        wf.setnchannels(self.channel)
        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
        wf.setframerate(self.frame_rate)
        wf.writeframes(b"".join(frames))
        wf.close()

    def audio_record_worker(self):
        print("Audio record worker start.")
        while self.is_listening:
            wave_data = self.recorder.read(self.chunk)
            self.wait_process_frames_queue.put(wave_data)
        print("Audio record worker end.")

    def start_audio_record_thread(self):
        self.start_recorder()
        self.is_listening = True
        t = threading.Thread(target=self.audio_record_worker)
        self.threads.append(t)
        t.setDaemon(True)
        t.start()

    def __speech_recognize(self, frames):
        frames_np = [
            np.frombuffer(x, np.int16).flatten().astype(np.float32) for x in frames
        ]

        audio = np.concatenate(frames_np, axis=0) / 32768.0
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.model, mel, options)
        return result

    def is_need_recognization(self):
        is_need_recognization = False
        for frame_status in self.cache_frames_status:
            is_need_recognization = is_need_recognization or frame_status.get(
                "is_speech"
            )
        return is_need_recognization

    def is_need_pause(self):
        if len(self.cache_frames) < self.pause_chunks_num_threshold:
            return False
        is_need_pause = True
        for i in range(self.pause_chunks_num_threshold):
            index = -i - 1
            is_need_pause = is_need_pause and (
                not self.cache_frames_status[index].get("is_speech")
            )
        return is_need_pause

    def __process_cache(self, is_pause=False):
        self.chunks_count = 0
        self.speech_content += self.cache_speech[-1]
        if is_pause:
            self.speech_content += self.pause_flag
        # self.save_wave(frames=self.cache_frames)
        # with open("output.txt", "w", encoding="utf-8") as f:
        #     f.write(self.speech_content)
        self.cache_frames = []
        self.cache_speech = []

    def run(self, imshow=False):
        self.start_audio_record_thread()
        self.chunks_count = 0

        while True:
            chunks_num = math.ceil(
                self.time_speed_last_recognization / self.time_speed_each_chunk
            )
            i = 0
            while True:
                wave_data = self.wait_process_frames_queue.get()
                #
                is_speech = self.__vad.is_speech(wave_data, sample_rate=self.frame_rate)
                self.cache_frames.append(wave_data)
                self.cache_frames_status.append({"is_speech": is_speech})

                if imshow:
                    wave_data_np = (
                        np.frombuffer(wave_data, np.int16).flatten().astype(np.float32)
                    )
                    plt.plot(
                        np.linspace(
                            len(wave_data_np) * self.chunks_count,
                            len(wave_data_np) * (self.chunks_count + 1),
                            len(wave_data_np),
                        ),
                        wave_data_np,
                        color="green" if is_speech else "grey",
                    )
                    plt.pause(0.01)

                self.chunks_count += 1
                i += 1
                if i >= chunks_num:
                    break

            if self.is_need_recognization():
                # speech recogization
                start_time_speech_recognization = time.time()
                result = self.__speech_recognize(frames=self.cache_frames)
                self.cache_speech.append(result.text)
                print(f"recog: {self.cache_speech[-1]}")
                self.time_speed_last_recognization = (
                    time.time() - start_time_speech_recognization
                )
            if self.is_need_pause():
                self.__process_cache(is_pause=True)

            if self.chunks_count >= self.cache_chunk_num:
                self.__process_cache()
        self.close_recorder()


if __name__ == "__main__":
    PROFILE_BEGIN_SESSION("project_init", "project_init_profile.json")
    app = WisperLoop()
    PROFILE_END_SESSION()
    PROFILE_BEGIN_SESSION("project_running", "project_running_profile.json")
    app.run()
    PROFILE_END_SESSION()

import whisper


class ModelCard:
    def __init__(self) -> None:
        self.download_root = "./model"
        pass

    @property
    def model(self):
        pass

    @property
    def sliding_dur(self):
        pass


class WhisperModelCard(ModelCard):
    def __init__(self):
        super().__init__()
        self.__sliding_dur = 30
        self.__model = whisper.load_model("tiny", download_root=self.download_root)
        # self.__model = whisper.load_model("tiny.en", download_root=self.download_root)
        # self.__model = whisper.load_model("base", download_root=self.download_root)
        # self.__model = whisper.load_model("base.en", download_root=self.download_root)
        # self.__model = whisper.load_model("small", download_root=self.download_root)
        # self.__model = whisper.load_model("small.en", download_root=self.download_root)
        # self.__model = whisper.load_model("medium", download_root=self.download_root)
        # self.__model = whisper.load_model("medium.en", download_root=self.download_root)
        # self.__model = whisper.load_model("large", download_root=self.download_root)
        # self.__model = whisper.load_model("large-v1", download_root=self.download_root)
        # self.__model = whisper.load_model("large-v2", download_root=self.download_root)

    @property
    def model(self):
        return self.__model

    @property
    def sliding_dur(self):
        return self.__sliding_dur

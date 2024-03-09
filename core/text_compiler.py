class TextCompiler:
    def __init__(self) -> None:
        self.__pause_flag = "#"

    @property
    def pause_flag(self):
        return self.__pause_flag

from abc import ABC, abstractmethod


class Source(ABC):
    @abstractmethod
    def get_frame():
        pass

    @abstractmethod
    def get_framecount():
        pass

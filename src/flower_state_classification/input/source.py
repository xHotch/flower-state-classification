from abc import ABC, abstractmethod


class Source(ABC):
    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def get_framecount(self):
        pass

    @abstractmethod
    def width(self):
        pass

    @abstractmethod
    def height(self):
        pass
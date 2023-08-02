from abc import ABC, abstractmethod


class Notifier(ABC):
    """Abstract class for notifiers."""

    @abstractmethod
    def notify(self, message: str) -> None:
        """Send a notification."""
        pass

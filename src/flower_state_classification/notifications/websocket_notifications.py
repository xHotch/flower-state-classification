import websockets
from flower_state_classification.notifications.notifier import Notifier
import asyncio
import logging

logger = logging.getLogger(__name__)


class WebsocketNotifier(Notifier):
    def __init__(self, websocket_host: str, websocket_port: str) -> None:
        super().__init__()
        self.uri = f"ws://{websocket_host}:{websocket_port}"

    def notify(self, message: str) -> None:
        asyncio.run(self._send_message(message))

    async def _send_message(self, message: str) -> None:
        try:
            async with websockets.connect(self.uri) as websocket:
                notification = {
                    "message": message,
                }
                await websocket.send(message)
        except OSError as e:
            logger.error(f"Could not connect to websocket server at {self.uri}")


def main():
    notifier = WebsocketNotifier("localhost", 8765)
    notifier.notify("test")


if __name__ == "__main__":
    main()

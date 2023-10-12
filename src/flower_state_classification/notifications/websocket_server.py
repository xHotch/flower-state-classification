import websockets

import logging


logger = logging.getLogger(__name__)


class WebsocketServer:
    """
    Class that runs a websocket server.
    """
    def __init__(self, websocket_host, websocket_port) -> None:
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port

    async def _handle_message(self, websocket, path) -> None:
        try:
            async for message in websocket:
                logger.info(f"Received message: {message}")
        except websockets.exceptions.ConnectionClosedError:
            logger.info("Connection closed")

    async def run(self):
        logging.info(f"Starting Websocket server at {self.websocket_host}:{self.websocket_port}")
        server = await websockets.serve(self._handle_message, self.websocket_host, self.websocket_port)
        logging.info(f"Websocket server started at {self.websocket_host}:{self.websocket_port}")

        await server.wait_closed()

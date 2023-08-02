import asyncio
from flower_state_classification.debug.settings import Settings
from flower_state_classification.notifications.notifier import Notifier
import telegram


class TelegramBot(Notifier):
    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token
        self.bot = telegram.Bot(token=self.token)
        asyncio.run(self.test())

    async def test(self):
        async with bot:
            print(await bot.get_me())

    def notify(self, message: str) -> None:
        return super().notify(message)


if __name__ == "__main__":
    settings = Settings()
    bot = TelegramBot(settings)

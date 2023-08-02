import asyncio
import logging
from flower_state_classification.debug.settings import Settings

def main():
    settings = Settings

    logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    server = settings.server
    asyncio.run(server.run())
    

if __name__ == "__main__":
    main()
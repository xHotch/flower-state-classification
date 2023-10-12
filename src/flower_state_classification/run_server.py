import asyncio
import logging
from flower_state_classification.settings.settings import Settings

"""
Simple script to run the server specified in the Settings file to read notifications from the pipeline.
"""

def main():
    settings = Settings

    logging.basicConfig(level=settings.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    server = settings.server
    asyncio.run(server.run())


if __name__ == "__main__":
    main()

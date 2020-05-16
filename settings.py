from dotenv import load_dotenv
from pathlib import Path  # python3 only
import os

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("CRYPTO_COMPARE_API_KEY")
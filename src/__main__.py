"""Entry point for `python -m src`."""
from src.main import RAGCLI
# pyrefly: ignore [missing-import]
import fire

if __name__ == "__main__":
    fire.Fire(RAGCLI)

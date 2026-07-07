"""Entry point for `python -m src`."""
from src.main import RAGCLI
import fire

if __name__ == "__main__":
    fire.Fire(RAGCLI)

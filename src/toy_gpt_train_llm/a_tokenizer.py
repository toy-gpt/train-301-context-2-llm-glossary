"""a_tokenizer.py - A simple tokenizer for toy language-model training.

This module reads text from a corpus file and converts it into a list of tokens.

Concepts:
- token: a discrete unit of text used by a model (in this project, a token is a word).
- tokenize: the process of splitting text into tokens.

Notes:
- This tokenizer uses whitespace splitting for clarity and inspectability.
- Real language models often use subword tokenizers (breaking a word into subparts).
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.a_tokenizer import SimpleTokenizer

__all__ = ["SimpleTokenizer", "CORPUS_DIR", "DEFAULT_CORPUS_PATH"]


LOG: logging.Logger = get_logger("TOKEN", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
CORPUS_DIR: Final[Path] = BASE_DIR / "corpus"
DEFAULT_CORPUS_PATH: Final[Path] = CORPUS_DIR / "010_llm_glossary.txt"


def main() -> None:
    """Demonstrate tokenization on the default corpus file."""
    import statistics

    log_header(LOG, "Tokenizer Demo")

    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    LOG.info(f"First 10 tokens: {tokens[:10]}")
    LOG.info(f"Total number of tokens: {len(tokens)}")

    if tokens:
        avg_token_length: float = statistics.mean(len(token) for token in tokens)
        LOG.info(f"Average token length: {avg_token_length:.2f}")
    else:
        LOG.info("No tokens available to calculate average length.")


if __name__ == "__main__":
    main()

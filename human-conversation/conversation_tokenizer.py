"""Train a byte-pair tokenizer and apply it to Conversation.csv."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


BASE_VOCAB_SIZE = 256
TEXT_COLUMNS = ("question", "answer")


class BytePairTokenizer:
    """Small byte-level BPE tokenizer with reversible UTF-8 encoding."""

    def __init__(self, merges: Iterable[tuple[int, int, int]] = ()) -> None:
        self.merges = list(merges)
        self._refresh_state()

    def _refresh_state(self) -> None:
        self.merge_lookup = {
            (left, right): token_id for left, right, token_id in self.merges
        }
        self.merge_rank = {
            (left, right): rank
            for rank, (left, right, _token_id) in enumerate(self.merges)
        }
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        for left, right, token_id in self.merges:
            self.vocab[token_id] = self.vocab[left] + self.vocab[right]

    @staticmethod
    def _merge_pair(
        token_ids: list[int], pair: tuple[int, int], new_token_id: int
    ) -> list[int]:
        merged = []
        index = 0
        while index < len(token_ids):
            if (
                index + 1 < len(token_ids)
                and token_ids[index] == pair[0]
                and token_ids[index + 1] == pair[1]
            ):
                merged.append(new_token_id)
                index += 2
            else:
                merged.append(token_ids[index])
                index += 1
        return merged

    def train(self, texts: Iterable[str], vocab_size: int = 512) -> None:
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_VOCAB_SIZE}")

        sequences = [list(text.encode("utf-8")) for text in texts if text]
        self.merges = []

        for token_id in range(BASE_VOCAB_SIZE, vocab_size):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))

            if not pair_counts:
                break

            # Counter preserves insertion order, making ties deterministic.
            pair, _count = pair_counts.most_common(1)[0]
            sequences = [
                self._merge_pair(sequence, pair, token_id) for sequence in sequences
            ]
            self.merges.append((pair[0], pair[1], token_id))

        self._refresh_state()

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        while len(token_ids) >= 2:
            pairs = set(zip(token_ids, token_ids[1:]))
            known_pairs = [pair for pair in pairs if pair in self.merge_lookup]
            if not known_pairs:
                break
            pair = min(known_pairs, key=self.merge_rank.__getitem__)
            token_ids = self._merge_pair(token_ids, pair, self.merge_lookup[pair])
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        encoded = b"".join(self.vocab[token_id] for token_id in token_ids)
        return encoded.decode("utf-8")

    def save(self, path: Path) -> None:
        model = {
            "type": "byte_pair_encoding",
            "base_vocab_size": BASE_VOCAB_SIZE,
            "vocab_size": len(self.vocab),
            "merges": self.merges,
        }
        path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BytePairTokenizer":
        model = json.loads(path.read_text(encoding="utf-8"))
        merges = [tuple(item) for item in model["merges"]]
        return cls(merges)


def read_conversations(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("The input CSV does not contain a header row")
        missing = [column for column in TEXT_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
        return reader.fieldnames, list(reader)


def tokenize_csv(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    vocab_size: int,
) -> None:
    fieldnames, rows = read_conversations(input_path)
    training_texts = [
        row[column] for row in rows for column in TEXT_COLUMNS if row[column]
    ]

    tokenizer = BytePairTokenizer()
    tokenizer.train(training_texts, vocab_size=vocab_size)

    original_token_count = 0
    encoded_token_count = 0
    encoded_rows = []

    for row in rows:
        encoded_row = dict(row)
        for column in TEXT_COLUMNS:
            text = row[column]
            token_ids = tokenizer.encode(text)
            if tokenizer.decode(token_ids) != text:
                raise ValueError(f"Round-trip validation failed in column {column}")
            encoded_row[column] = " ".join(map(str, token_ids))
            original_token_count += len(text.encode("utf-8"))
            encoded_token_count += len(token_ids)
        encoded_rows.append(encoded_row)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(encoded_rows)

    tokenizer.save(model_path)

    reduction = (
        100 * (original_token_count - encoded_token_count) / original_token_count
        if original_token_count
        else 0
    )
    print(f"Rows tokenized: {len(rows)}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Tokens before BPE: {original_token_count}")
    print(f"Tokens after BPE: {encoded_token_count}")
    print(f"Token reduction: {reduction:.2f}%")
    print(f"Tokenized CSV: {output_path}")
    print(f"Tokenizer model: {model_path}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "Conversation.csv",
        help="Source conversation CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "Conversation_tokenized.csv",
        help="Destination CSV containing token IDs",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "conversation_tokenizer.json",
        help="Destination for the learned merge rules",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Total vocabulary size, including the 256 byte tokens",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenize_csv(args.input, args.output, args.model, args.vocab_size)


if __name__ == "__main__":
    main()

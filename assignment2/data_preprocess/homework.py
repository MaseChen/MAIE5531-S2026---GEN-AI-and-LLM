import argparse
import html
from html.parser import HTMLParser
import re
import requests
import json
from utils import read_warc_file, read_wet_file
from datasets import load_dataset
from typing import Set, Dict
import string


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {
            "p",
            "br",
            "div",
            "li",
            "tr",
            "section",
            "article",
            "header",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data:
            self._chunks.append(data)

    def get_text(self) -> str:
        text = "".join(self._chunks)
        text = html.unescape(text)
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()


def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open("./bad_word_list.txt", "r") as file:
        records = file.read().strip().split("\n")
        bad_words = [record.lower() for record in records]
        return set(bad_words)


def html_to_text(html) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    if html is None:
        return ""

    if isinstance(html, bytes):
        decoded = html.decode("utf-8", errors="ignore")
    else:
        decoded = str(html)

    extractor = _HTMLTextExtractor()
    extractor.feed(decoded)
    extractor.close()
    return extractor.get_text()


def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    # Replace US social security numbers (XXX-XX-XXXX format)
    if not text:
        return text

    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "XXX-XX-XXXX", text)
    text = re.sub(r"(?<!\d)\+1\d{10}\b", lambda _: "+" + "X" * 11, text)
    return text


def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """
    if not text:
        return ""

    cleaned_paragraphs: list[str] = []
    for paragraph in text.split("\n"):
        stripped = paragraph.strip()
        if not stripped:
            continue
        if not any(char in string.punctuation for char in stripped):
            continue
        if re.search(r"\b\w{101,}\b", stripped):
            continue
        cleaned_paragraphs.append(stripped)

    return "\n".join(cleaned_paragraphs)


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    if not text or not text.strip():
        return False

    lower_text = text.lower()
    for bad_word in retrieve_bad_words():
        if bad_word and bad_word in lower_text:
            return False

    if not any(char in string.punctuation for char in text):
        return False

    allowed_chars = sum(
        char.isalnum() or char.isspace() or char in string.punctuation for char in text
    )
    if allowed_chars / max(len(text), 1) < 0.8:
        return False

    return True


def is_english_text(text: str) -> bool:
    """Detects if text is primarily in English based on character distribution.
    Args:
        text (str): Text to analyze
    Returns:
        bool: True if text is primarily English, False otherwise
    """
    if not text or not text.strip():
        return False

    ascii_alpha_chars = sum(char.isascii() and char.isalpha() for char in text)
    if not text:
        return False

    return ascii_alpha_chars / len(text) >= 0.6


def deduplicate_texts(texts: list[str]) -> list[str]:
    """Deduplicates text by removing duplicate sentences.
    Args:
        texts (list[str]): List of text strings to deduplicate.
    Returns:
        list[str]: Deduplicated list of texts. Implemented a simple Jaccard similarity based deduplication.
    """

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def jaccard_similarity(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    deduplicated_texts: list[str] = []
    deduplicated_tokens: list[set[str]] = []
    similarity_threshold = 0.5

    for text in texts:
        tokens = tokenize(text)
        is_duplicate = False
        for existing_tokens in deduplicated_tokens:
            if jaccard_similarity(tokens, existing_tokens) >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated_texts.append(text)
            deduplicated_tokens.append(tokens)

    return deduplicated_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", type=str, default="", help="Specify the path for your warc file."
    )
    parser.add_argument(
        "--dfname",
        type=str,
        default="",
        help="Specify the path where you stored topic_dataset.json",
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=30,
        help="Specify the number of records you want to parse (only used for debugging with smaller sets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cleaned_documents.txt",
        help="Output file for cleaned text documents",
    )
    # parser.add_argument('--wet_name', type = str, default = '', help = 'Specify the path for your wet file.')
    args = parser.parse_args()

    if args.fname:
        seen = 0
        passes = 0

        with open(args.output, "w", encoding="utf-8") as output_file:
            for url, html_text in read_warc_file(args.fname, args.num_records):
                seen += 1
                # print("Before HTML to text: ", str(html_text))
                text = html_to_text(html_text)
                # print("\n\n\nAfter HTML to text: ", text)
                cleaned_text = clean_text(text)
                # print("After cleaning: ", cleaned_text)
                cleaned_nopii_text = replace_pii(cleaned_text)
                # print("After PII removal: ", cleaned_nopii_text)
                passes_check = heuristic_quality_filter(cleaned_nopii_text)
                is_english = is_english_text(cleaned_nopii_text)
                print(url)
                print("Passes heuristic quality filter:", passes_check)
                print("Is English text:", is_english)
                if passes_check and is_english:
                    passes += 1
                    # Replace newlines with spaces to keep each document on one line
                    single_line_text = (
                        cleaned_nopii_text.replace("\n", " ").replace("\r", " ").strip()
                    )
                    output_file.write(single_line_text + "\n")
                    print("Saved cleaned English document to output file")
                elif passes_check and not is_english:
                    print("Document filtered out: not English")

        print(f"{passes} passed out of {seen} records processed.")
        print(f"Cleaned documents saved to: {args.output}")

    if args.dfname:
        with open(args.dfname, "r") as f:
            raw_texts = json.load(f)
        raw_texts = [item["text"] for item in raw_texts["data"]]
        deduplicated_texts = deduplicate_texts(raw_texts)
        print(
            f"{len(deduplicated_texts)} deduplicated out of {len(raw_texts)} records processed."
        )
    else:
        print("Usage: python homework.py --fname data.warc")

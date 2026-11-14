import html
import re


def clean_text(text: str) -> str:
    """
    Normalise news text before feature extraction.

    - Decode HTML entities.
    - Normalise whitespaces
    - Convert text to lowercase
    """

    if not isinstance(text, str):
        return ""

    # Decode HTML entities such as &#39; and &quot;
    text = html.unescape(text)

    # Normalise whitespaces
    text = re.sub(r"\s+", " ", text)

    # Standardise case
    text = text.lower().strip()

    return text

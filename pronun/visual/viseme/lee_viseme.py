"""Mode B: Linguistically defined visemes via Lee's Viseme Map.

No training needed — visemes are predefined from phoneme-to-viseme mapping.
"""

from pronun.audio.g2p import text_to_visemes
from pronun.data.lee_map import LEE_VISEME_LABELS, NUM_VISEMES, arpabet_to_viseme


class LeeViseme:
    """Lee's linguistically defined viseme mapper."""

    def __init__(self):
        self.num_visemes = NUM_VISEMES
        self.labels = LEE_VISEME_LABELS

    def text_to_viseme_sequence(self, text: str) -> list[int]:
        """Convert text to a viseme ID sequence using Lee's map.

        Args:
            text: English text to convert.

        Returns:
            List of viseme IDs.
        """
        return text_to_visemes(text)

    def phoneme_to_viseme(self, arpabet_phoneme: str) -> int:
        """Convert a single ARPAbet phoneme to viseme ID."""
        return arpabet_to_viseme(arpabet_phoneme)

    def viseme_label(self, viseme_id: int) -> str:
        """Get human-readable label for a viseme ID."""
        return self.labels.get(viseme_id, f"UNK_{viseme_id}")

    def describe_sequence(self, viseme_ids: list[int]) -> list[str]:
        """Convert viseme ID sequence to label sequence."""
        return [self.viseme_label(v) for v in viseme_ids]

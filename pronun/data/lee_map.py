"""Lee's 13-viseme mapping (Lee & Yook, 2002).

Maps ARPAbet phonemes to viseme IDs based on visual similarity
of mouth shapes during articulation.
"""

LEE_VISEME_LABELS = {
    0: "SIL",
    1: "P",
    2: "T",
    3: "K",
    4: "CH",
    5: "F",
    6: "W",
    7: "IY",
    8: "EY",
    9: "AA",
    10: "AO",
    11: "UH",
    12: "ER",
}

# ARPAbet phoneme (without stress digits) → viseme ID
ARPABET_TO_VISEME = {
    # Silence
    "SIL": 0,
    "SP": 0,
    "SPN": 0,

    # Viseme 1 — P: bilabial stops + nasal
    "P": 1,
    "B": 1,
    "M": 1,

    # Viseme 2 — T: alveolar consonants
    "T": 2,
    "D": 2,
    "S": 2,
    "Z": 2,
    "TH": 2,
    "DH": 2,

    # Viseme 3 — K: velar/glottal + alveolar nasal/lateral/glide
    "K": 3,
    "G": 3,
    "NG": 3,
    "N": 3,
    "L": 3,
    "Y": 3,
    "HH": 3,

    # Viseme 4 — CH: postalveolar affricates/fricatives
    "CH": 4,
    "JH": 4,
    "SH": 4,
    "ZH": 4,

    # Viseme 5 — F: labiodental fricatives
    "F": 5,
    "V": 5,

    # Viseme 6 — W: approximants (lip rounding)
    "R": 6,
    "W": 6,

    # Viseme 7 — IY: high front vowels
    "IY": 7,
    "IH": 7,

    # Viseme 8 — EY: mid front vowels
    "EY": 8,
    "EH": 8,
    "AE": 8,

    # Viseme 9 — AA: low/open vowels + diphthongs
    "AA": 9,
    "AW": 9,
    "AY": 9,
    "AH": 9,

    # Viseme 10 — AO: back rounded vowels + diphthongs
    "AO": 10,
    "OY": 10,
    "OW": 10,

    # Viseme 11 — UH: high back vowels
    "UH": 11,
    "UW": 11,

    # Viseme 12 — ER: r-colored vowels
    "ER": 12,
    "AX": 12,
}

NUM_VISEMES = 13


def arpabet_to_viseme(phoneme: str) -> int:
    """Convert an ARPAbet phoneme to its Lee viseme ID.

    Strips stress digits (0, 1, 2) from vowels before lookup.
    Returns viseme 0 (SIL) for unknown phonemes.
    """
    clean = phoneme.rstrip("012")
    return ARPABET_TO_VISEME.get(clean, 0)

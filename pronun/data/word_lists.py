"""Built-in practice word lists organized by difficulty and phoneme focus."""

BEGINNER_WORDS = [
    "hello", "thank", "please", "water", "sorry",
    "happy", "money", "coffee", "chicken", "teacher",
    "student", "morning", "evening", "number", "color",
]

INTERMEDIATE_WORDS = [
    "breakfast", "bathroom", "beautiful", "important", "together",
    "different", "exercise", "yesterday", "chocolate", "vegetable",
    "comfortable", "understand", "restaurant", "interesting", "appreciate",
]

ADVANCED_WORDS = [
    "pronunciation", "environment", "unfortunately", "communication", "opportunity",
    "responsibility", "enthusiastic", "particularly", "extraordinary", "organization",
    "archaeological", "characteristic", "representative", "sophisticated", "unbelievable",
]

# Words grouped by target phoneme contrast (common for Mandarin speakers)
PHONEME_FOCUS = {
    "r_vs_l": ["right", "light", "red", "led", "rock", "lock", "rain", "lane"],
    "th_voiced": ["the", "this", "that", "them", "there", "those", "other", "brother"],
    "th_voiceless": ["think", "three", "throw", "thing", "thought", "through", "thank", "teeth"],
    "v_vs_w": ["vine", "wine", "vest", "west", "vet", "wet", "very", "wary"],
    "b_vs_p": ["bat", "pat", "big", "pig", "bay", "pay", "bet", "pet"],
    "short_vowels": ["bit", "bet", "bat", "but", "bot", "put", "sit", "set"],
}

ALL_WORDS = BEGINNER_WORDS + INTERMEDIATE_WORDS + ADVANCED_WORDS

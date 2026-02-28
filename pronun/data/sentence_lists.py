"""Built-in practice sentence lists organized by difficulty and phoneme focus."""

BEGINNER_SENTENCES = [
    "How are you today",
    "Thank you very much",
    "I like this coffee",
    "Where is the bathroom",
    "Please sit down here",
    "The weather is nice",
    "I need some water",
    "Can you help me",
    "Good morning teacher",
    "See you tomorrow",
    "I am very happy",
    "This is my house",
    "What is your name",
    "I want to go",
    "Let me think about",
]

INTERMEDIATE_SENTENCES = [
    "I would like to order breakfast please",
    "The restaurant is on the other side",
    "She is studying at the university",
    "We need to leave before it rains",
    "Can you tell me where the library is",
    "I have been living here for three years",
    "The children are playing in the garden",
    "He drives to work every morning",
    "This exercise is really good for you",
    "I appreciate your help with the project",
    "The movie was better than I expected",
    "We should finish this before tomorrow",
    "Do you understand what he is saying",
    "The vegetables at the market are fresh",
    "I need to make a phone call",
]

ADVANCED_SENTENCES = [
    "The pronunciation of these words requires careful practice",
    "Unfortunately the representative was not available",
    "Communication is particularly important in this organization",
    "The archaeological discovery was truly extraordinary",
    "She demonstrated an enthusiastic and sophisticated approach",
    "Environmental responsibility is everyone's opportunity",
    "The characteristic of this material is unbelievable",
    "His thorough explanation cleared up the misunderstanding",
    "The temperature dropped dramatically throughout the evening",
    "We thoroughly investigated the mathematical relationship",
    "Their determination led to a remarkable achievement",
    "The photographer captured an absolutely breathtaking image",
    "International collaboration strengthens our communities",
    "The manufacturing process requires considerable precision",
    "She effortlessly articulated the philosophical argument",
]

# Sentences grouped by target phoneme contrast
SENTENCE_FOCUS = {
    "r_vs_l": [
        "The red light turned right on the lane",
        "Larry really likes the rolling river",
        "She locked the rock collection in the room",
        "The railroad leads to a lovely little lake",
    ],
    "th_voiced": [
        "The other brother lives with them there",
        "This is the one that they gathered together",
        "Whether the weather is good or bad",
        "Those brothers are smoother than the others",
    ],
    "th_voiceless": [
        "I think three thousand is too much",
        "She threw the ball through the teeth of the wind",
        "The thief thought nothing of the theft",
        "Both math and health require thorough thinking",
    ],
    "v_vs_w": [
        "The very warm wine was in the vest",
        "We went to view the vast western valley",
        "The vet was worried about the vivid wound",
        "Victor wanted to wave to the village woman",
    ],
    "b_vs_p": [
        "The big pig broke past the barn",
        "Pat bought a beautiful purple basket",
        "The baby played by the open pool",
        "Bob put the brown paper in the box",
    ],
    "short_vowels": [
        "The man in the red hat sat on a bus",
        "She bit into the fresh bread and butter",
        "His cup of hot coffee sat on the desk",
        "Put the book on the big black shelf",
    ],
}

ALL_SENTENCES = BEGINNER_SENTENCES + INTERMEDIATE_SENTENCES + ADVANCED_SENTENCES

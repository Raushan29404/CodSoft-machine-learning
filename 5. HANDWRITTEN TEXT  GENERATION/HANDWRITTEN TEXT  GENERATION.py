# ==========================================
# HANDWRITTEN TEXT GENERATION
# Character Level RNN (LSTM)
# ==========================================

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

print("Loading Dataset...")

# ==========================================
# LOAD TEXT FILE
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
text_file = os.path.join(BASE_DIR, "text_data.txt")

if not os.path.exists(text_file):
    print("ERROR: text_data.txt not found!")
    exit()

with open(text_file, "r", encoding="utf-8") as f:
    text = f.read().lower()

# Check empty file
if len(text.strip()) == 0:
    print("ERROR: text_data.txt is empty!")
    print("Open text_data.txt and add some text.")
    exit()

print("Dataset Length:", len(text))

# ==========================================
# CHARACTER MAPPING
# ==========================================

chars = sorted(list(set(text)))

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

n_chars = len(text)
n_vocab = len(chars)

print("Total Characters:", n_chars)
print("Unique Characters:", n_vocab)

# ==========================================
# CREATE SEQUENCES
# ==========================================

seq_length = 50

X = []
y = []

for i in range(0, n_chars - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]

    X.append([char_to_int[c] for c in seq_in])
    y.append(char_to_int[seq_out])

if len(X) == 0:
    print("ERROR: Not enough text data.")
    print("Add more text inside text_data.txt")
    exit()

print("Total Patterns:", len(X))

# ==========================================
# RESHAPE
# ==========================================

X = np.reshape(X, (len(X), seq_length, 1))
X = X / float(n_vocab)

y = to_categorical(y)

# ==========================================
# BUILD MODEL
# ==========================================

model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(
    Dense(
        y.shape[1],
        activation="softmax"
    )
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Model...")

model.fit(
    X,
    y,
    epochs=5,
    batch_size=64,
    verbose=1
)

# ==========================================
# GENERATE TEXT
# ==========================================

start = np.random.randint(0, len(X)-1)

pattern = X[start]

generated_text = ""

print("\nGenerated Text:\n")

for i in range(300):

    x = np.reshape(pattern, (1, len(pattern), 1))

    prediction = model.predict(x, verbose=0)

    index = np.argmax(prediction)

    result = int_to_char[index]

    generated_text += result

    pattern = np.append(pattern, [[index]], axis=0)
    pattern = pattern[1:]

print(generated_text)

print("\nProgram Finished Successfully!")
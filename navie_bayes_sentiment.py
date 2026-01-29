import math

# =========================
# DATASET
# label: 0 = negatif, 1 = netral, 2 = positif
# masing-masing 2 data (VERSI 2)
# =========================
data = [
    # POSITIF
    ("pembelajarannya jelas dan sangat membantu pemahaman", 2),
    ("fitur platform lengkap dan nyaman digunakan", 2),

    # NEGATIF
    ("penjelasan materi kurang jelas dan membingungkan", 0),
    ("aplikasi sering lambat dan mengganggu proses belajar", 0),

    # NETRAL
    ("materi yang diberikan cukup standar", 1),
    ("platform bisa digunakan meskipun tidak terlalu istimewa", 1)
]

# =========================
# STOPWORD & STEMMING SEDERHANA
# =========================
stopwords = ["dan", "yang", "di", "ke", "dari", "buat", "aja", "lah", "tapi", "gak"]

def stemming(word):
    for suf in ["nya", "lah", "an", "in"]:
        if word.endswith(suf):
            return word.replace(suf, "")
    return word

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    text = text.lower()
    text = text.replace(".", "").replace(",", "")
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords]
    tokens = [stemming(w) for w in tokens]
    return tokens

# =========================
# PERSIAPAN DATA
# =========================
classes = set(label for _, label in data)

texts = []
labels = []

for text, label in data:
    texts.append(preprocess(text))
    labels.append(label)

# =========================
# HITUNG PRIOR
# =========================
prior = {}
for c in classes:
    prior[c] = labels.count(c) / len(labels)

# =========================
# HITUNG FREKUENSI KATA
# =========================
word_freq = {c: {} for c in classes}
class_word_count = {c: 0 for c in classes}

for tokens, label in zip(texts, labels):
    for w in tokens:
        word_freq[label][w] = word_freq[label].get(w, 0) + 1
        class_word_count[label] += 1

# =========================
# VOCABULARY
# =========================
vocab = set()
for c in word_freq:
    vocab.update(word_freq[c])

vocab_size = len(vocab)

# =========================
# FUNGSI PREDIKSI (NAIVE BAYES)
# =========================
def predict(text):
    tokens = preprocess(text)
    scores = {}

    for c in classes:
        score = math.log(prior[c])
        for w in tokens:
            count = word_freq[c].get(w, 0) + 1
            score += math.log(count / (class_word_count[c] + vocab_size))
        scores[c] = score

    return max(scores, key=scores.get)

# =========================
# INTERFACE
# =========================
print("Model Naive Bayes siap digunakan (Versi 2)")

while True:
    kalimat = input("\nMasukkan kalimat (exit untuk keluar): ")
    if kalimat.lower() == "exit":
        break

    hasil = predict(kalimat)

    if hasil == 0:
        print("Sentimen: NEGATIF")
    elif hasil == 1:
        print("Sentimen: NETRAL")
    else:
        print("Sentimen: POSITIF")

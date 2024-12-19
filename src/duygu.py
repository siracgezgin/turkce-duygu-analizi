import re
from jpype import startJVM, shutdownJVM, isJVMStarted
from zemberek.morphology import TurkishMorphology
from zemberek.tokenization import TurkishTokenizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ZEMBEREK_PATH = "C:/Users/sirac/OneDrive/Masaüstü/pro/zemberek-full.jar"

def load_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def load_phrases(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def clean_text(text):
    # Sorunlu karakterleri temizle
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def handle_negation(tokens):
    negation_words = {"değil", "yok", "hiç", "hiçbir", "istemiyorum", "sevmedim", "sevmiyorum", "beğenmedi"}
    for i, word in enumerate(tokens):
        if word in negation_words:
            if i > 0:
                tokens[i-1] = f"NOT_{tokens[i-1]}"  # Negation effect on previous word
    return tokens

def analyze_sentiment(sentence):
    try:
        raw_tokens = TurkishTokenizer.DEFAULT.tokenize(sentence)
        tokens = [t.content.lower() for t in raw_tokens if t.content.strip() != '']
        tokens = handle_negation(tokens)

        # Bigrams and Trigrams extraction using CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(2, 3))  # Extract bigrams and trigrams
        token_string = " ".join(tokens)
        ngrams = vectorizer.fit_transform([token_string])
        bigrams_trigrams = vectorizer.get_feature_names_out()

        # Add bigrams and trigrams to the token list
        tokens.extend(bigrams_trigrams)

        results = morphology.analyze_and_disambiguate(" ".join(tokens)).best_analysis()

        score = 0
        has_negation_word = any(word in NEGATION_WORDS for word in tokens)

        for result in results:
            root = result.item.root if result.item.root else result.item.normalized_form
            root = root.lower() if root else ""

            if root in POSITIVE_WORDS:
                score += 1
            elif root in NEGATIVE_WORDS:
                score -= 1

        for bg in bigrams_trigrams:
            if bg in POSITIVE_PHRASES:
                score += 1
            elif bg in NEGATIVE_PHRASES:
                score -= 1

        if has_negation_word:
            score = -score

        return "Pozitif" if score > 0 else "Negatif"

    except Exception as e:
        print(f"Hata oluştu: {e}")
        return "Hata"

try:
    # Zemberek JVM'i başlatma
    startJVM("-Djava.class.path=" + ZEMBEREK_PATH)
    morphology = TurkishMorphology.create_with_defaults()

    # TXT dosyalarından kelime ve ifade verilerini yükleme
    STOPWORDS = load_words("C:/Users/sirac/OneDrive/Masaüstü/pro/stopwords.txt")
    POSITIVE_WORDS = load_words("C:/Users/sirac/OneDrive/Masaüstü/pro/positive_words.txt")
    NEGATIVE_WORDS = load_words("C:/Users/sirac/OneDrive/Masaüstü/pro/negative_words.txt")
    POSITIVE_PHRASES = load_phrases("C:/Users/sirac/OneDrive/Masaüstü/pro/positive_phrases.txt")
    NEGATIVE_PHRASES = load_phrases("C:/Users/sirac/OneDrive/Masaüstü/pro/negative_phrases.txt")
    
    NEGATION_WORDS = {"değil", "yok", "hiç", "hiçbir", "istemiyorum", "sevmedim", "sevmiyorum", "beğenmedi"}

    print("Lütfen analiz etmek istediğiniz cümleleri girin. Çıkmak için 'q' yazın.")
    while True:
        user_input = input("Cümle: ")
        if user_input.strip().lower() == 'q':
            break
        if user_input.strip():
            cleaned_text = clean_text(user_input)
            tahmin = analyze_sentiment(cleaned_text)
            print(f"\n🔍 Cümle: {user_input}")
            print(f"🔍 Analiz Sonucu: {tahmin}\n")

    # Test verilerini oku ve değerlendirme yap
    df = pd.read_excel("C:/Users/sirac/OneDrive/Masaüstü/pro/larger_balanced_dataset.xlsx")
    test_data = list(zip(df["Cümle"], df["Sınıf"]))

    DP = 0
    DN = 0
    YP = 0
    YN = 0

    for cümle, gerçek_etiket in test_data:
        temiz_cümle = clean_text(cümle)
        tahmin = analyze_sentiment(temiz_cümle)

        if tahmin == "Pozitif" and gerçek_etiket == "Pozitif":
            DP += 1
        elif tahmin == "Negatif" and gerçek_etiket == "Negatif":
            DN += 1
        elif tahmin == "Pozitif" and gerçek_etiket == "Negatif":
            YP += 1
        elif tahmin == "Negatif" and gerçek_etiket == "Pozitif":
            YN += 1

    toplam = DP + DN + YP + YN
    doğruluk = (DP + DN) / toplam if toplam > 0 else 0
    kesinlik = DP / (DP + YP) if (DP + YP) > 0 else 0
    anma = DP / (DP + YN) if (DP + YN) > 0 else 0
    f1 = (2 * kesinlik * anma) / (kesinlik + anma) if (kesinlik + anma) > 0 else 0

    print("Performans Metrikleri:")
    print(f"Doğruluk: {doğruluk:.2f}")
    print(f"Kesinlik (Precision): {kesinlik:.2f}")
    print(f"Anma (Recall): {anma:.2f}")
    print(f"F1 Skoru: {f1:.2f}")

except Exception as e:
    print(f"Hata oluştu: {e}")
finally:
    if isJVMStarted():
        shutdownJVM()
        print("JVM kapatıldı.")
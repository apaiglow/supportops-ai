from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase = True, ngram_range = (1, 2), min_df = 2, sublinear_tf = True)
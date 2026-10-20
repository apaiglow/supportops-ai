from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from supportops.preprocessing import build_tfidf_vectorizer
from sklearn.svm import LinearSVC

def build_intent_classifier() -> Pipeline:

    pipeline = Pipeline(steps = [
        ('tf-idf', build_tfidf_vectorizer()),
        ('classifier', LogisticRegression(max_iter = 1000, random_state = 42))
    ])
    return pipeline

def build_linear_svc_classifier() -> Pipeline:
    return Pipeline(steps = [
    ('tf-idf', build_tfidf_vectorizer()),
    ('classifier', LinearSVC(random_state = 43, max_iter = 5000))
    ])
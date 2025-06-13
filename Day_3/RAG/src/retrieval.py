from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [chunk[0] for chunk in chunks]
        self.vectorizer = TfidfVectorizer().fit(self.texts)
        self.vectors = self.vectorizer.transform(self.texts)

    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors)[0]
        ranked_indices = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in ranked_indices]

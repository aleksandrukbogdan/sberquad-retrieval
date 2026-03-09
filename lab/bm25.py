import math


class BM25:
    # Lucene BM25, запросные дубли не усиливают терм.
    # idf = log(1 + (N - df + 0.5) / (df + 0.5))
    # tf' = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    def __init__(self, docs):
        self.n = len(docs)
        self.dl = [len(doc) for doc in docs]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        df = {}
        postings = {}
        tfs = []
        for i, doc in enumerate(docs):
            tf = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            tfs.append(tf)
            for token in tf:
                df[token] = df.get(token, 0) + 1
                postings.setdefault(token, []).append(i)
        self.tf = tfs
        self.postings = postings
        self.idf = {
            token: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def scores(self, query, k1, b):
        if self.avgdl == 0:
            return {}
        seen = set()
        acc = {}
        for token in query:
            if token in seen:
                continue
            seen.add(token)
            idf = self.idf.get(token)
            if idf is None:
                continue
            for i in self.postings[token]:
                tf = self.tf[i][token]
                denom = tf + k1 * (1.0 - b + b * self.dl[i] / self.avgdl)
                acc[i] = acc.get(i, 0.0) + idf * (tf * (k1 + 1.0)) / denom
        return acc


def top_docs(scores, ids, topn):
    order = sorted(scores, key=lambda i: (-scores[i], ids[i]))
    return [ids[i] for i in order[:topn]]

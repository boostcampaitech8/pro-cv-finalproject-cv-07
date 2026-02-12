import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_valid_date_range(t, date_list, window_size=5):
    idx = date_list.index(t)
    
    start_idx = max(0, idx - window_size)
    valid_dates = date_list[start_idx:idx]
    
    return valid_dates


def mmr(query_emb, doc_embs, lambda_=0.7, top_k=5):
    selected = []
    candidates = list(range(len(doc_embs)))

    query_emb = np.array(query_emb).reshape(1, -1)
    doc_embs = np.array(doc_embs)

    while len(selected) < top_k and candidates:
        scores = []
        for i in candidates:
            sim_to_query = cosine_similarity(query_emb, doc_embs[i:i+1])[0][0]
            sim_to_selected = max(
                cosine_similarity(doc_embs[i:i+1], doc_embs[selected])[0]
                if selected else [0]
            )
            score = lambda_ * sim_to_query - (1 - lambda_) * sim_to_selected
            scores.append(score)

        best = candidates[np.argmax(scores)]
        selected.append(best)
        candidates.remove(best)

    return selected
import os
import csv
import math
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel


# =========================
# LOAD MODEL
# =========================
def load_model(model_dir, base_model='sentence-transformers/all-MiniLM-L6-v2'):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(model_dir)

    model.eval()
    model.to(device)

    return tokenizer, model


# =========================
# BATCH ENCODING
# =========================
def encode_batch(texts, tokenizer, model, batch_size=256):
    device = next(model.parameters()).device
    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding queries"):
            batch = texts[i:i+batch_size]

            inputs = tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                padding=True
            ).to(device)

            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)

            all_embs.append(emb)

    return torch.cat(all_embs, dim=0)


# =========================
# LOAD DATA
# =========================
def load_playlist_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_playlist_tracks(items_csv, tracks_csv):
    track_meta = {}

    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Track metadata"):
            track_meta[row["track_uri"]] = (
                row["track_name"],
                row["artist_name"]
            )

    playlist_tracks = {}

    with open(items_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Playlist tracks"):
            pid = row["pid"].strip()
            uri = row["track_uri"]

            if uri in track_meta:
                playlist_tracks.setdefault(pid, []).append(track_meta[uri])

    return playlist_tracks


# =========================
# METRICS
# =========================
def compute_metrics(recommended_songs, relevant_songs, top_n):
    G_T = set(relevant_songs)
    G_A = set(a for _, a in relevant_songs)

    R = len(G_T)

    hits = sum(1 for s in recommended_songs[:top_n] if s in G_T)
    hit_score = hits / min(top_n, R) if R > 0 else 0.0

    precision = hits / top_n if top_n > 0 else 0.0
    recall = hits / R if R > 0 else 0.0

    mrr = 0.0
    for i, s in enumerate(recommended_songs[:top_n]):
        if s in G_T:
            mrr = 1 / (i + 1)
            break

    top_r = recommended_songs[:R]
    S_T = set(top_r)
    S_A = set(a for _, a in top_r)

    exact = S_T & G_T
    artist = S_A & G_A

    r_precision = (len(exact) + 0.25 * len(artist)) / R if R > 0 else 0.0

    rel = [1 if s in G_T else 0 for s in recommended_songs[:top_n]]

    def dcg(r):
        return sum(v / math.log2(i + 2) for i, v in enumerate(r))

    idcg = dcg(sorted(rel, reverse=True))
    ndcg = dcg(rel) / idcg if idcg > 0 else 0.0

    return hit_score, precision, recall, mrr, r_precision, ndcg


# =========================
# MAIN
# =========================
def main():

    model_dir = "/content/drive/MyDrive/playlist_project/models/cross_entropy_model"
    emb_file = "/content/drive/MyDrive/playlist_project/embeddings/playlists_embeddings_cross_entropy.pkl"
    items_csv = "/content/drive/MyDrive/playlist_project/playlist_continuation_data/csvs/items.csv"
    tracks_csv = "/content/drive/MyDrive/playlist_project/playlist_continuation_data/csvs/tracks.csv"
    clusters_test_csv = "/content/drive/MyDrive/playlist_project/clustering-no-split/split/represented/clusters_test.csv"

    out10 = "/content/drive/MyDrive/playlist_project/results/evaluation_cross_entropy_10.csv"
    out66 = "/content/drive/MyDrive/playlist_project/results/evaluation_cross_entropy_66.csv"
    out100 = "/content/drive/MyDrive/playlist_project/results/evaluation_cross_entropy_100.csv"

    tokenizer, model = load_model(model_dir)

    # ---------- Load embeddings ----------
    playlist_embeddings = load_playlist_embeddings(emb_file)

    pids = list(playlist_embeddings.keys())
    all_embs = torch.tensor(
        np.stack([playlist_embeddings[pid]["embedding"] for pid in pids]),
        dtype=torch.float32,
        device="cuda"
    )
    all_embs = F.normalize(all_embs, dim=1)

    # ---------- Load tracks ----------
    playlist_tracks = load_playlist_tracks(items_csv, tracks_csv)

    # ---------- Load test data ----------
    test_names, test_pids, cluster_ids = [], [], []

    with open(clusters_test_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_names.append(row["Playlist Title"].strip())
            test_pids.append(row["Playlist ID"].strip())
            cluster_ids.append(row["Cluster ID"])

    # ---------- Encode queries ----------
    query_embs = encode_batch(test_names, tokenizer, model)
    query_embs = F.normalize(query_embs, dim=1)

    # ---------- Chunked similarity ----------
    TOP_K = 50
    QUERY_BATCH = 256

    results_10, results_66, results_100 = [], [], []

    for start in tqdm(range(0, query_embs.size(0), QUERY_BATCH), desc="Similarity batches"):
        end = start + QUERY_BATCH
        q_batch = query_embs[start:end]

        sim = torch.matmul(q_batch, all_embs.T)
        _, topk_idx = torch.topk(sim, k=TOP_K, dim=1)
        topk_idx = topk_idx.cpu()

        for i in range(topk_idx.size(0)):
            global_idx = start + i

            indices = topk_idx[i].numpy()
            similar_pids = [pids[idx] for idx in indices]

            counter = Counter()
            for pid in similar_pids:
                for track in playlist_tracks.get(str(pid), []):
                    counter[track] += 1

            top_songs = [song for song, _ in counter.most_common(500)]
            relevant = list(set(playlist_tracks.get(test_pids[global_idx], [])))

            for k, store in zip(
                [10, 66, 100],
                [results_10, results_66, results_100]
            ):
                hit, p, r, mrr, rp, ndcg = compute_metrics(top_songs, relevant, k)

                store.append({
                    "Cluster ID": cluster_ids[global_idx],
                    "Playlist ID": test_pids[global_idx],
                    "Playlist Title": test_names[global_idx],
                    f"HIT@{k}": hit,
                    f"Precision@{k}": p,
                    f"Recall@{k}": r,
                    f"MRR@{k}": mrr,
                    "R-Precision": rp,
                    f"NDCG@{k}": ndcg
                })

        del sim, topk_idx
        torch.cuda.empty_cache()

    # ---------- Save function ----------
    def save_csv(path, results, k):
        fieldnames = [
            "Cluster ID", "Playlist ID", "Playlist Title",
            f"HIT@{k}", f"Precision@{k}", f"Recall@{k}",
            f"MRR@{k}", "R-Precision", f"NDCG@{k}"
        ]
        with open(path, 'w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    save_csv(out10, results_10, 10)
    save_csv(out66, results_66, 66)
    save_csv(out100, results_100, 100)

    print("Saved all 3 CSVs.")


if __name__ == "__main__":
    main()
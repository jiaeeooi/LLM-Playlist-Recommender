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
import itertools

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
    """
    Compute all metrics including HIT@N, Precision@N, Recall@N, MRR@N, 
    R-Precision (adjusted for top_n), and NDCG@N.
    """
    G_T = set(relevant_songs)
    G_A = set(a for _, a in relevant_songs)
    R = len(G_T)

    # HIT@N
    hits = sum(1 for s in recommended_songs[:top_n] if s in G_T)
    hit_score = hits / min(top_n, R) if R > 0 else 0.0

    # Precision & Recall
    precision = hits / top_n if top_n > 0 else 0.0
    recall = hits / R if R > 0 else 0.0

    # MRR@N
    mrr = 0.0
    for i, s in enumerate(recommended_songs[:top_n]):
        if s in G_T:
            mrr = 1 / (i + 1)
            break

    # R-Precision adjusted: use min(R, top_n)
    top_r = recommended_songs[:min(R, top_n)]
    S_T = set(top_r)
    S_A = set(a for _, a in top_r)
    exact = S_T & G_T
    artist = S_A & G_A
    r_precision = (len(exact) + 0.25 * len(artist)) / R if R > 0 else 0.0

    # NDCG@N
    rel = [1 if s in G_T else 0 for s in recommended_songs[:top_n]]
    def dcg(r):
        return sum(v / math.log2(i + 2) for i, v in enumerate(r))
    idcg = dcg(sorted(rel, reverse=True))
    ndcg = dcg(rel) / idcg if idcg > 0 else 0.0

    return hit_score, precision, recall, mrr, r_precision, ndcg


# =========================
# Aggregation Methods
# =========================
def combsum(track_scores):
    """Sum of track scores across playlists"""
    counter = Counter()
    for track, score in track_scores:
        counter[track] += score
    return counter

def combmnz(track_scores, playlist_counts):
    """CombMNZ = CombSUM * number of playlists a track appears in"""
    counter = combsum(track_scores)
    for track in counter:
        counter[track] *= playlist_counts[track]
    return counter

def bordafuse(track_scores, playlist_rankings):
    """BordaFuse: points inversely proportional to rank"""
    counter = Counter()
    for ranking in playlist_rankings:
        L = len(ranking)
        for pos, track in enumerate(ranking):
            counter[track] += (L - pos)
    return counter

def logisr(track_scores):
    """LogISR: dampen high scores using log"""
    counter = Counter()
    for track, score in track_scores:
        counter[track] += math.log(1 + score)
    return counter

def aggregate_tracks(similar_pids, playlist_tracks, playlist_scores_norm, top_n_list=[10, 66, 500]):
    """
    Aggregate tracks using different unsupervised methods.
    Returns a dictionary: method -> top_n -> list of track lists per query
    """
    results_per_method = {k: {N: [] for N in top_n_list} 
                          for k in ['combsum', 'combmnz', 'bordafuse', 'logisr']}

    for q_idx, top_pids in enumerate(tqdm(similar_pids, desc="Aggregating tracks per query")):
        # --- prepare track-level features ---
        track_scores = []   # list of (track, score)
        playlist_counts = Counter()  # number of playlists a track appears in
        playlist_rankings = []  # for BordaFuse

        for pid in top_pids:
            score = playlist_scores_norm[pid]  # normalized cosine similarity
            tracks = playlist_tracks.get(pid, [])
            track_scores.extend([(track, score) for track, _ in tracks])
            playlist_counts.update([track for track, _ in tracks])
            playlist_rankings.append([track for track, _ in tracks])

        # --- aggregate tracks ---
        agg_combsum = combsum(track_scores)
        agg_combmnz = combmnz(track_scores, playlist_counts)
        agg_borda = bordafuse(track_scores, playlist_rankings)
        agg_logisr = logisr(track_scores)

        # --- sort top tracks ---
        top_combsum = [t for t, _ in agg_combsum.most_common(max(top_n_list))]
        top_combmnz = [t for t, _ in agg_combmnz.most_common(max(top_n_list))]
        top_borda = [t for t, _ in agg_borda.most_common(max(top_n_list))]
        top_logisr = [t for t, _ in agg_logisr.most_common(max(top_n_list))]

        # --- store top-N for each method ---
        for N in top_n_list:
            results_per_method['combsum'][N].append(top_combsum[:N])
            results_per_method['combmnz'][N].append(top_combmnz[:N])
            results_per_method['bordafuse'][N].append(top_borda[:N])
            results_per_method['logisr'][N].append(top_logisr[:N])

    return results_per_method

# =========================
# TWRA RERANK FUNCTION
# =========================
def twra_rerank(f_rank_scores, playlist_counts, lambda_val=0.5):
    """
    Two-Way Ranking Aggregation (TWRA)
    
    Args:
        f_rank_scores: list of tuples (track, score), higher score = more relevant
        playlist_counts: dict {track: frequency across candidate playlists}
        lambda_val: float between 0 and 1, weight of backward rank

    Returns:
        sorted_tracks: list of tracks sorted by ag-rank ascending
    """
    # 1. Compute f-rank: rank by descending f_rank_scores
    f_sorted = sorted(f_rank_scores, key=lambda x: -x[1])
    f_rank_dict = {track: rank for rank, (track, _) in enumerate(f_sorted, start=1)}

    # 2. Compute b-rank: rank by descending frequency (higher freq = rank 1)
    sorted_tracks_by_freq = sorted(playlist_counts.items(), key=lambda x: -x[1])
    b_rank_dict = {track: rank for rank, (track, _) in enumerate(sorted_tracks_by_freq, start=1)}

    # 3. Compute aggregated rank
    ag_rank_dict = {}
    for track, _ in f_rank_scores:
        f_r = f_rank_dict.get(track, len(f_rank_scores)+1)
        b_r = b_rank_dict.get(track, len(f_rank_scores)+1)
        ag_rank_dict[track] = (1 - lambda_val) * f_r + lambda_val * b_r

    # 4. Sort by aggregated rank ascending
    sorted_tracks = sorted(ag_rank_dict, key=lambda t: ag_rank_dict[t])
    return sorted_tracks

# =========================
# HAMMING DISTANCE DIVERSITY
# =========================
def hamming_distance_diversity(recommendations):
    """
    recommendations: list of lists, each sublist is top-L tracks for a playlist
    returns: average pairwise Hamming distance (diversity)
    """
    n = len(recommendations)
    L = len(recommendations[0])
    total_hd = 0
    count = 0

    for i in range(n):
        for j in range(i+1, n):
            common = len(set(recommendations[i]) & set(recommendations[j]))
            hd_ij = 1 - (common / L)
            total_hd += hd_ij
            count += 1

    return total_hd / count if count > 0 else 0.0

# =========================
# GINI COEFFICIENT FOR COVERAGE
# =========================
def gini_coverage(recommendations):
    """
    recommendations: list of lists, each sublist is top-L tracks for a playlist
    returns: Gini coefficient measuring imbalance in track recommendation frequency
    """
    track_counts = Counter()
    for rec in recommendations:
        for t in rec:
            track_counts[t] += 1

    freqs = np.array(sorted(track_counts.values()))  # ascending
    n = len(freqs)
    if n <= 1:
        return 0.0

    # compute Gini coefficient
    cumulative = np.cumsum(freqs)
    gini = 1 - (2 / (n - 1)) * np.sum((np.arange(1, n+1) - 1) * freqs / cumulative[-1])
    return gini



# =========================
# MAIN
# =========================
def main():

    model_dir = "/content/drive/MyDrive/playlist_project/models/cross_entropy_model"
    emb_file = "/content/drive/MyDrive/playlist_project/embeddings/playlists_embeddings_cross_entropy.pkl"
    #model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    #emb_file = "/content/drive/MyDrive/playlist_project/embeddings/playlists_embeddings_pretrained.pkl"

    items_csv = "/content/drive/MyDrive/playlist_project/playlist_continuation_data/csvs/items.csv"
    tracks_csv = "/content/drive/MyDrive/playlist_project/playlist_continuation_data/csvs/tracks.csv"
    clusters_test_csv = "/content/drive/MyDrive/playlist_project/clustering-no-split/split/represented/clusters_test.csv"

    out_dir = "/content/drive/MyDrive/playlist_project/results/"
    out_files = {10: "evaluation_cross_entropy_10.csv",
                 66: "evaluation_cross_entropy_66.csv",
                 500: "evaluation_cross_entropy_500.csv"}
                 
    #out10 = "/content/drive/MyDrive/playlist_project/results/evaluation_cross_entropy_10.csv"
    #out66 = "/content/drive/MyDrive/playlist_project/results/evaluation_cross_entropy_66.csv"
    #out500 = "/content/drive/MyDrive/playlist_project/results/evaluation_cross_entropy_500.csv"
    #out10 = "/content/drive/MyDrive/playlist_project/results/evaluation_pretrained_10.csv"
    #out66 = "/content/drive/MyDrive/playlist_project/results/evaluation_pretrained_66.csv"
    #out500 = "/content/drive/MyDrive/playlist_project/results/evaluation_pretrained_500.csv"

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
    top_n_list = [10, 66, 500]

    # Store all results per method
    results_all = {method: {N: [] for N in top_n_list} 
                   for method in ['original', 'combsum', 'combmnz', 'bordafuse', 'logisr', 'twra']}

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

            '''
            counter = Counter()
            for pid in similar_pids:
                for track in playlist_tracks.get(str(pid), []):
                    counter[track] += 1

            top_songs = [song for song, _ in counter.most_common(500)]
            relevant = list(set(playlist_tracks.get(test_pids[global_idx], [])))
            '''

            # ---------- Prepare track candidates ----------
            candidates = {}
            playlist_scores_norm = {}  # normalized similarity per playlist
            playlist_rankings = []     # for BordaFuse

            for pid in similar_pids:
                score = sim[i, pids.index(pid)].item()
                playlist_scores_norm[pid] = score
                tracks = playlist_tracks.get(str(pid), [])
                playlist_rankings.append([track for track, _ in tracks])
                for track in tracks:
                    if track not in candidates:
                        candidates[track] = {"playlist_score": score, "frequency": 1}
                    else:
                        candidates[track]["frequency"] += 1
                        candidates[track]["playlist_score"] = max(candidates[track]["playlist_score"], score)
            
            # normalize playlist_score and frequency 0-1
            max_score = max(c["playlist_score"] for c in candidates.values()) or 1
            max_freq = max(c["frequency"] for c in candidates.values()) or 1
            for c in candidates.values():
                c["playlist_score_norm"] = c["playlist_score"] / max_score
                c["frequency_norm"] = c["frequency"] / max_freq

            # ---------- Original Top-K ----------
            counter_orig = Counter()
            for pid in similar_pids:
                for track in playlist_tracks.get(str(pid), []):
                    counter_orig[track] += 1
            top_original = [t for t, _ in counter_orig.most_common(max(top_n_list))]

             # ---------- CombSUM ----------
            track_scores = [(track, c["playlist_score_norm"]) for track, c in candidates.items()]
            agg_combsum = combsum(track_scores)
            top_combsum = [t for t, _ in agg_combsum.most_common(max(top_n_list))]

            # ---------- CombMNZ ----------
            playlist_counts = {track: c["frequency"] for track, c in candidates.items()}
            agg_combmnz = combmnz(track_scores, playlist_counts)
            top_combmnz = [t for t, _ in agg_combmnz.most_common(max(top_n_list))]

            # ---------- BordaFuse ----------
            agg_borda = bordafuse(track_scores, playlist_rankings)
            top_borda = [t for t, _ in agg_borda.most_common(max(top_n_list))]

            # ---------- LogISR ----------
            agg_logisr = logisr(track_scores)
            top_logisr = [t for t, _ in agg_logisr.most_common(max(top_n_list))]

            # ---------- TWRA ----------
            top_twra = twra_rerank(track_scores, playlist_counts, lambda_val=0.5)

            # ---------- Compute metrics ----------
            relevant = list(set(playlist_tracks.get(test_pids[global_idx], [])))
            for N in top_n_list:
                for method, top_songs in zip(
                    ['original', 'combsum', 'combmnz', 'bordafuse', 'logisr', 'twra'],
                    [top_original, top_combsum, top_combmnz, top_borda, top_logisr, top_twra]
                ):
                    topN = top_songs[:N]
                    hit, p, r, mrr, rp, ndcg = compute_metrics(topN, relevant, N)
                    results_all[method][N].append({
                        "Cluster ID": cluster_ids[global_idx],
                        "Playlist ID": test_pids[global_idx],
                        "Playlist Title": test_names[global_idx],
                        f"HIT@{N}": hit,
                        f"Precision@{N}": p,
                        f"Recall@{N}": r,
                        f"MRR@{N}": mrr,
                        "R-Precision": rp,
                        f"NDCG@{N}": ndcg
                    })

        del sim, topk_idx
        torch.cuda.empty_cache()

    # ---------- Save CSVs ----------
    for N in top_n_list:
        for method in results_all:
            path = os.path.join(out_dir, f"{method}_{N}.csv")
            fieldnames = [
                "Cluster ID", "Playlist ID", "Playlist Title",
                f"HIT@{N}", f"Precision@{N}", f"Recall@{N}",
                f"MRR@{N}", "R-Precision", f"NDCG@{N}"
            ]
            with open(path, 'w', newline='', encoding='utf8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_all[method][N])

    print("Saved all CSVs.")


if __name__ == "__main__":
    main()
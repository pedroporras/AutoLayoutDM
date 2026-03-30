# -*- coding: utf-8 -*-
"""LayoutDM (preprocesamiento iter1).ipynb

"""

# ===========================================
# RICO -> LayoutDM tokens builder (full script)
# ===========================================
# - Walk semantic_annotations/
# - Parse JSON -> continuous (category, x,y,w,h normalized)
# - Compute stats to choose M
# - Split train/val/test
# - Fit KMeans on train only for x/y/w/h
# - Build tokens [N,M,5] with PAD
# - Export tokens_*.pt + metadata files
#
# No filtering (takes all nodes with valid bounds)
# Correctness > speed
# ===========================================

import os
import json
import math
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

# sklearn is commonly available in Colab
from sklearn.cluster import KMeans

# -----------------------------
# Config
# -----------------------------
RICO_SEMANTIC_DIR = "/content/semantic_annotations/semantic_annotations"  # <-- CHANGE THIS
OUT_DIR = "/content/layoutdm_rico_tokens"                # output folder

SEED = 42

# Default bins for x/y/w/h (KMeans clusters)
BINS = 64  # Reasonable default as explained above.

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Percentile used to choose M
M_PERCENTILE = 95  # choose M as p95 of element counts
MIN_AREA = 0.0     # "take all": we keep 0.0; still we reject w<=0 or h<=0
DROP_ROOT = True   # do not include root node itself as element

# KMeans settings
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

# Optional: to avoid insane memory, sample values for KMeans if huge
KMEANS_SAMPLE_LIMIT = 2_000_000  # total samples per modality max (x or y etc.)

# -----------------------------
# Utils: parsing RICO semantic JSON
# -----------------------------
def _walk_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = [node]
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            out.extend(_walk_nodes(child))
    return out

def _normalize_bounds(b: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """
    Tries to interpret bounds as either:
      A) [x0, y0, x1, y1]
      B) [x, y, w, h]   (common mismatch)
    Returns (x0,y0,x1,y1) or None.
    """
    if not b or len(b) != 4:
        return None
    try:
        x0, y0, a, c = map(float, b)
    except Exception:
        return None

    # First assume it's [x0,y0,x1,y1]
    x1, y1 = a, c
    w = x1 - x0
    h = y1 - y0

    # If degenerate, try interpret as [x,y,w,h]
    if w <= 0 or h <= 0:
        w2, h2 = a, c
        if w2 > 0 and h2 > 0:
            x1 = x0 + w2
            y1 = y0 + h2
            w = x1 - x0
            h = y1 - y0

    if w <= 0 or h <= 0:
        return None
    return x0, y0, x1, y1

def _infer_screen_size_from_tree(data: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Infer (sx0, sy0, sx1, sy1) as the global bbox over all nodes with valid bounds.
    This rescues files where root bounds are invalid/missing.
    """
    nodes = _walk_nodes(data)
    mins = [float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf")]
    found = False

    for n in nodes:
        b = n.get("bounds")
        if not isinstance(b, list):
            continue
        nb = _normalize_bounds(b)
        if nb is None:
            continue
        x0, y0, x1, y1 = nb
        mins[0] = min(mins[0], x0)
        mins[1] = min(mins[1], y0)
        maxs[0] = max(maxs[0], x1)
        maxs[1] = max(maxs[1], y1)
        found = True

    if not found:
        return None

    sx0, sy0, sx1, sy1 = mins[0], mins[1], maxs[0], maxs[1]
    if (sx1 - sx0) <= 0 or (sy1 - sy0) <= 0:
        return None
    return sx0, sy0, sx1, sy1

def _bounds_to_xywh_norm(bounds: List[float], screen_w: float, screen_h: float) -> Optional[Tuple[float, float, float, float]]:
    nb = _normalize_bounds(bounds)
    if nb is None:
        return None
    x0, y0, x1, y1 = nb

    w_px = x1 - x0
    h_px = y1 - y0
    if screen_w <= 0 or screen_h <= 0:
        return None
    if w_px <= 0 or h_px <= 0:
        return None

    x_c = (x0 + x1) / 2.0
    y_c = (y0 + y1) / 2.0

    x = x_c / screen_w
    y = y_c / screen_h
    w = w_px / screen_w
    h = h_px / screen_h

    # clamp defensive
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x, y, w, h

def infer_base_wh_from_root(bounds):
    """
    Heurística simple y efectiva para RICO:
    si root es un sub-rectángulo del full screen, entonces
    full_w ~= x0 + x1, full_h ~= y0 + y1
    """
    nb = _normalize_bounds(bounds)
    if nb is None:
        return None
    x0, y0, x1, y1 = nb
    base_w = x0 + x1
    base_h = y0 + y1
    if base_w <= 0 or base_h <= 0:
        return None
    return float(base_w), float(base_h)

def rico_semantic_json_to_elements(data: Dict[str, Any]) -> Tuple[Tuple[float, float], List[Dict[str, Any]]]:
    """
    Robust version:
      - Accepts root bounds as [x0,y0,x1,y1] OR [x,y,w,h]
      - If root invalid, infers screen bbox from all nodes
    """
    root_bounds = data.get("bounds")
    if not isinstance(root_bounds, list):
        raise ValueError("Root has no bounds")

    # ✅ base_w/base_h para normalización (full screen coords)
    base = infer_base_wh_from_root(root_bounds)
    if base is None:
        # fallback: infer from tree bbox (menos ideal pero funciona)
        screen_bbox = _infer_screen_size_from_tree(data)
        if screen_bbox is None:
            raise ValueError("Cannot infer screen size: no valid bounds in tree.")
        sx0, sy0, sx1, sy1 = screen_bbox
        base_w = float(sx1 - sx0)
        base_h = float(sy1 - sy0)
    else:
        base_w, base_h = base

    nodes = _walk_nodes(data)
    if DROP_ROOT and nodes:
        nodes = nodes[1:]

    elements = []
    for n in nodes:
        b = n.get("bounds")
        if not isinstance(b, list):
            continue
        nb = _normalize_bounds(b)
        if nb is None:
            continue
        x0, y0, x1, y1 = nb

        w_px = x1 - x0
        h_px = y1 - y0
        if w_px <= 0 or h_px <= 0:
            continue

        x_c = (x0 + x1) / 2.0
        y_c = (y0 + y1) / 2.0

        # ✅ NORMALIZACIÓN CORRECTA A FULL SCREEN
        x = x_c / base_w
        y = y_c / base_h
        w = w_px / base_w
        h = h_px / base_h

        # clamp defensivo
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        category = n.get("componentLabel") or n.get("class") or "UNKNOWN"
        elements.append({"category": str(category), "x": x, "y": y, "w": w, "h": h})

    return (base_w, base_h), elements
    # root_bounds = data.get("bounds")
    # screen_bbox = None

    # if isinstance(root_bounds, list):
    #     nb = _normalize_bounds(root_bounds)
    #     if nb is not None:
    #         screen_bbox = nb

    # if screen_bbox is None:
    #     screen_bbox = _infer_screen_size_from_tree(data)

    # if screen_bbox is None:
    #     raise ValueError("Cannot infer screen size: no valid bounds in tree.")

    # sx0, sy0, sx1, sy1 = screen_bbox
    # screen_w = float(sx1 - sx0)
    # screen_h = float(sy1 - sy0)
    # if screen_w <= 0 or screen_h <= 0:
    #     raise ValueError("Invalid screen size after inference.")

    # nodes = _walk_nodes(data)
    # if DROP_ROOT and nodes:
    #     nodes = nodes[1:]

    # elements = []
    # for n in nodes:
    #     b = n.get("bounds")
    #     if not isinstance(b, list):
    #         continue
    #     xywh = _bounds_to_xywh_norm(b, screen_w, screen_h)
    #     if xywh is None:
    #         continue

    #     x, y, w, h = xywh
    #     if (w * h) < MIN_AREA:
    #         continue

    #     category = n.get("componentLabel") or n.get("class") or "UNKNOWN"
    #     elements.append({"category": str(category), "x": x, "y": y, "w": w, "h": h})

    # return (screen_w, screen_h), elements


# -----------------------------
# Build dataset: load all screens -> continuous frames
# -----------------------------
def load_all_screens(semantic_dir: str) -> List[Dict[str, Any]]:
    files = sorted([f for f in os.listdir(semantic_dir) if f.lower().endswith(".json")])
    if not files:
        raise RuntimeError(f"No .json files found in: {semantic_dir}")

    screens = []
    bad = []
    for fname in files:
        path = os.path.join(semantic_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            (W, H), elements = rico_semantic_json_to_elements(data)
            screens.append({
                "id": os.path.splitext(fname)[0],
                "screen_w": W,
                "screen_h": H,
                "elements": elements,
            })
        except Exception as e:
            # correctness > speed: don't crash whole run, record failures
            bad.append((fname, str(e)))

    if bad:
        print(f"\n[WARN] {len(bad)} files could not be parsed into screens. Showing first 20:")
        for fn, err in bad[:20]:
            print(" -", fn, "->", err)

    if not screens:
        raise RuntimeError("All files failed parsing; check semantic_annotations content/format.")

    return screens


# -----------------------------
# Stats for M + choose M
# -----------------------------
def describe_counts(counts: List[int]) -> Dict[str, float]:
    arr = np.array(counts, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "n": float(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def choose_M_from_counts(counts: List[int], percentile: int = 95) -> int:
    arr = np.array(counts, dtype=np.float32)
    m = int(math.ceil(np.percentile(arr, percentile)))
    return max(1, m)


# -----------------------------
# Splitting
# -----------------------------
def split_ids(n: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]

    return train_idx, val_idx, test_idx


# -----------------------------
# Category mapping
# -----------------------------
def build_cat2id_from_train(train_screens: List[Dict[str, Any]]) -> Dict[str, int]:
    cats = set()
    for s in train_screens:
        for e in s["elements"]:
            cats.add(e["category"])
    cats = sorted(list(cats))
    return {c: i for i, c in enumerate(cats)}


# -----------------------------
# KMeans fitting and binning
# -----------------------------
def _collect_values_for_modality(screens: List[Dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for s in screens:
        for e in s["elements"]:
            vals.append(e[key])
    arr = np.array(vals, dtype=np.float32)
    return arr


def _maybe_subsample(arr: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if arr.size <= limit:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=limit, replace=False)
    return arr[idx]


def fit_kmeans_1d(values: np.ndarray, n_clusters: int, seed: int) -> Tuple[np.ndarray, KMeans]:
    """
    Fit 1D KMeans. Returns sorted centroids (ascending) and the fitted model.
    """
    # KMeans expects 2D
    X = values.reshape(-1, 1)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    km.fit(X)

    centers = km.cluster_centers_.reshape(-1)
    # Sort centers; we will use nearest-center assignment ourselves
    centers_sorted = np.sort(centers)
    return centers_sorted.astype(np.float32), km


def assign_to_nearest_centroid(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    values: [N] float
    centroids: [B] float sorted
    returns: [N] int in [0..B-1]
    """
    # For simplicity and correctness: brute-force with broadcasting.
    # (Not the fastest, but clear.)
    # distances: [N, B]
    distances = np.abs(values.reshape(-1, 1) - centroids.reshape(1, -1))
    return distances.argmin(axis=1).astype(np.int64)


# -----------------------------
# Build tokens for a split
# -----------------------------
def build_tokens_for_screens(
    screens: List[Dict[str, Any]],
    M: int,
    cat2id: Dict[str, int],
    centroids: Dict[str, np.ndarray],
    bins: int,
) -> torch.LongTensor:
    """
    Returns tokens: LongTensor [N, M, 5] where each row: (c_id, x_id, y_id, w_id, h_id)
    Tokens are discrete and PAD is applied.

    Special token ids:
      - For category: C classes => mask_id=C, pad_id=C+1
      - For bins: B => mask_id=B, pad_id=B+1
    """
    modalities = ["c", "x", "y", "w", "h"]
    N = len(screens)
    C = len(cat2id)

    # Special IDs
    c_mask_id = C
    c_pad_id = C + 1

    b_mask_id = bins
    b_pad_id = bins + 1

    tokens = torch.empty((N, M, 5), dtype=torch.long)

    for i, s in enumerate(screens):
        elems = s["elements"]

        # If more than M, truncate (simple rule: keep first M as-is)
        elems = elems[:M]

        # Build arrays for vectorized centroid assignment
        # Categories
        c_ids = []
        xs, ys, ws, hs = [], [], [], []

        for e in elems:
            c_ids.append(cat2id.get(e["category"], None))
            xs.append(e["x"])
            ys.append(e["y"])
            ws.append(e["w"])
            hs.append(e["h"])

        # Convert to numpy
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        ws = np.array(ws, dtype=np.float32)
        hs = np.array(hs, dtype=np.float32)

        # Assign bins
        x_ids = assign_to_nearest_centroid(xs, centroids["x"]) if xs.size else np.array([], dtype=np.int64)
        y_ids = assign_to_nearest_centroid(ys, centroids["y"]) if ys.size else np.array([], dtype=np.int64)
        w_ids = assign_to_nearest_centroid(ws, centroids["w"]) if ws.size else np.array([], dtype=np.int64)
        h_ids = assign_to_nearest_centroid(hs, centroids["h"]) if hs.size else np.array([], dtype=np.int64)

        # Fill tokens for real elements
        n_real = len(elems)
        if n_real > 0:
            # categories: if unknown category (shouldn't happen if cat2id built on train),
            # map to PAD? Here we map unknown to last valid category 0; but normally none.
            c_arr = np.array([ci if ci is not None else 0 for ci in c_ids], dtype=np.int64)

            tokens[i, :n_real, 0] = torch.from_numpy(c_arr)
            tokens[i, :n_real, 1] = torch.from_numpy(x_ids)
            tokens[i, :n_real, 2] = torch.from_numpy(y_ids)
            tokens[i, :n_real, 3] = torch.from_numpy(w_ids)
            tokens[i, :n_real, 4] = torch.from_numpy(h_ids)

        # Fill PAD for remaining slots
        if n_real < M:
            tokens[i, n_real:, 0] = c_pad_id
            tokens[i, n_real:, 1] = b_pad_id
            tokens[i, n_real:, 2] = b_pad_id
            tokens[i, n_real:, 3] = b_pad_id
            tokens[i, n_real:, 4] = b_pad_id

    return tokens

def decode_tokens_to_xywh(tokens_row, centroids, vocab_meta):
    """
    tokens_row: [M,5] tensor
    returns list of (c_id, x,y,w,h) floats for non-pad
    """
    import torch
    c_pad = vocab_meta["c"]["pad_id"]
    b_pad = vocab_meta["x"]["pad_id"]

    out = []
    for t in tokens_row:
        c_id, x_id, y_id, w_id, h_id = map(int, t.tolist())
        if c_id == c_pad:
            continue
        if x_id == b_pad or y_id == b_pad or w_id == b_pad or h_id == b_pad:
            continue
        x = float(centroids["x"][x_id])
        y = float(centroids["y"][y_id])
        w = float(centroids["w"][w_id])
        h = float(centroids["h"][h_id])
        out.append((c_id, x,y,w,h))
    return out

def sanity_check_decoded(train_tokens, centroids, vocab_meta, n=10):
    import random
    idxs = random.sample(range(train_tokens.shape[0]), k=min(n, train_tokens.shape[0]))
    for i in idxs:
        boxes = decode_tokens_to_xywh(train_tokens[i], centroids, vocab_meta)
        ok = 0
        for _,x,y,w,h in boxes:
            # caja centrada: corners
            x0, x1 = x - w/2, x + w/2
            y0, y1 = y - h/2, y + h/2
            if 0 <= x0 <= 1 and 0 <= y0 <= 1 and 0 <= x1 <= 1 and 0 <= y1 <= 1:
                ok += 1
        print("sample", i, "boxes", len(boxes), "inside", ok)

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading screens from:", RICO_SEMANTIC_DIR)
    screens = load_all_screens(RICO_SEMANTIC_DIR)
    print("Total screens:", len(screens))

    counts = [len(s["elements"]) for s in screens]
    stats = describe_counts(counts)
    print("\nElement count stats:", json.dumps(stats, indent=2))

    M = choose_M_from_counts(counts, percentile=M_PERCENTILE)
    print(f"\nChosen M (p{M_PERCENTILE}): {M}")

    # Split
    train_idx, val_idx, test_idx = split_ids(
        n=len(screens),
        seed=SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

    train_screens = [screens[i] for i in train_idx]
    val_screens = [screens[i] for i in val_idx]
    test_screens = [screens[i] for i in test_idx]

    print("\nSplit sizes:",
          "train", len(train_screens),
          "val", len(val_screens),
          "test", len(test_screens))

    # Category mapping from TRAIN only
    cat2id = build_cat2id_from_train(train_screens)
    C = len(cat2id)
    print("\nNum categories (train):", C)

    # Collect x/y/w/h values from TRAIN only for KMeans
    xs = _collect_values_for_modality(train_screens, "x")
    ys = _collect_values_for_modality(train_screens, "y")
    ws = _collect_values_for_modality(train_screens, "w")
    hs = _collect_values_for_modality(train_screens, "h")

    # Optional subsample for KMeans stability and memory
    xs_fit = _maybe_subsample(xs, KMEANS_SAMPLE_LIMIT, SEED + 1)
    ys_fit = _maybe_subsample(ys, KMEANS_SAMPLE_LIMIT, SEED + 2)
    ws_fit = _maybe_subsample(ws, KMEANS_SAMPLE_LIMIT, SEED + 3)
    hs_fit = _maybe_subsample(hs, KMEANS_SAMPLE_LIMIT, SEED + 4)

    print(f"\nFitting KMeans (train only) with BINS={BINS}")
    print("Train samples used for KMeans:",
          "x", xs_fit.size, "y", ys_fit.size, "w", ws_fit.size, "h", hs_fit.size)

    centroids = {}
    centroids["x"], _ = fit_kmeans_1d(xs_fit, BINS, SEED + 10)
    centroids["y"], _ = fit_kmeans_1d(ys_fit, BINS, SEED + 11)
    centroids["w"], _ = fit_kmeans_1d(ws_fit, BINS, SEED + 12)
    centroids["h"], _ = fit_kmeans_1d(hs_fit, BINS, SEED + 13)

    # Build tokens for each split
    print("\nBuilding tokens...")
    train_tokens = build_tokens_for_screens(train_screens, M, cat2id, centroids, BINS)
    val_tokens = build_tokens_for_screens(val_screens, M, cat2id, centroids, BINS)
    test_tokens = build_tokens_for_screens(test_screens, M, cat2id, centroids, BINS)

    # Save artifacts
    train_path = os.path.join(OUT_DIR, "tokens_train.pt")
    val_path = os.path.join(OUT_DIR, "tokens_val.pt")
    test_path = os.path.join(OUT_DIR, "tokens_test.pt")

    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)
    torch.save(test_tokens, test_path)

    # Save centroids for decoding
    torch.save(torch.from_numpy(centroids["x"]), os.path.join(OUT_DIR, "centroids_x.pt"))
    torch.save(torch.from_numpy(centroids["y"]), os.path.join(OUT_DIR, "centroids_y.pt"))
    torch.save(torch.from_numpy(centroids["w"]), os.path.join(OUT_DIR, "centroids_w.pt"))
    torch.save(torch.from_numpy(centroids["h"]), os.path.join(OUT_DIR, "centroids_h.pt"))

    # Save cat2id
    with open(os.path.join(OUT_DIR, "cat2id.json"), "w", encoding="utf-8") as f:
        json.dump(cat2id, f, ensure_ascii=False, indent=2)

    # Build vocab_meta for training loop
    # Special IDs:
    #   category: vocab = C + 2, mask_id = C, pad_id = C+1
    #   x/y/w/h : vocab = B + 2, mask_id = B, pad_id = B+1
    vocab_meta = {
        "c": {"vocab_size": C + 2, "mask_id": C, "pad_id": C + 1},
        "x": {"vocab_size": BINS + 2, "mask_id": BINS, "pad_id": BINS + 1},
        "y": {"vocab_size": BINS + 2, "mask_id": BINS, "pad_id": BINS + 1},
        "w": {"vocab_size": BINS + 2, "mask_id": BINS, "pad_id": BINS + 1},
        "h": {"vocab_size": BINS + 2, "mask_id": BINS, "pad_id": BINS + 1},
        "M": M,
        "bins": BINS,
        "seed": SEED,
        "split": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "M_percentile": M_PERCENTILE,
    }
    with open(os.path.join(OUT_DIR, "vocab_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, ensure_ascii=False, indent=2)

    print("\nSaved outputs to:", OUT_DIR)
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    print(" - vocab_meta.json, cat2id.json, centroids_*.pt")

    # Quick sanity prints
    print("\nSanity check:")
    print("train_tokens shape:", tuple(train_tokens.shape), "dtype:", train_tokens.dtype)
    print("Example first row (first 3 elements):\n", train_tokens[0, :3, :])

    sanity_check_decoded(train_tokens, centroids, vocab_meta)

# Clean up any previously downloaded corrupted files
!rm -f semantic_annotations.zip semantic_annotations.zip

# Use the direct download link for Google Cloud Storage
!wget -O semantic_annotations.zip https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip?alt=media

import zipfile

with zipfile.ZipFile('semantic_annotations.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/semantic_annotations')

tmp = os.listdir("/content/semantic_annotations/semantic_annotations")

# tmp2 = os.listdir("/content/rico_dataset/combined")

# sorted([int(x.split('.')[0]) for x in tmp2])[:5]

sorted([int(x.split('.')[0]) for x in tmp])[:5]

# tmp2[:5]

# if __name__ == "__main__":
screens = main()

!wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz

# prompt: untar unique_uis.tar.gz inside a folder unique_uis
!rm -rf rico_dataset
!rm -rf unique_uis
!mkdir rico_dataset
!tar -xzf unique_uis.tar.gz -C rico_dataset

ls rico_dataset/combined | wc -l

# from google.colab import drive
# drive.mount('/content/drive')

import os, json, random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -----------------------
# Paths (AJUSTA SI HACE FALTA)
# -----------------------
OUT_DIR = "/content/layoutdm_rico_tokens"  # donde guardaste tokens + centroids + meta
SEM_DIR = "/content/semantic_annotations/semantic_annotations"  # tu semantic dir real
# Candidatos donde podrían estar las imágenes (RICO suele tener these):
IMG_DIR_CANDIDATES = [
    "/content/rico/combined",
    "/content/rico/screenshots",
    "/content/rico/dataset/combined",
    "/content/rico_dataset/combined",
    "/content/rico_dataset/screenshots",
    "/content/screenshots",
    "/content/combined",
]

OVERLAY_OUT = os.path.join(OUT_DIR, "debug_overlays")
os.makedirs(OVERLAY_OUT, exist_ok=True)

K = 24               # cantidad de muestras a renderizar
MAX_BOXES = 55       # normalmente M
DRAW_LABELS = False  # si True, escribe el id de categoría en cada caja (más lento/ensucia)
SEED = 42

# -----------------------
# Load artifacts
# -----------------------
with open(os.path.join(OUT_DIR, "vocab_meta.json"), "r", encoding="utf-8") as f:
    vocab_meta = json.load(f)

with open(os.path.join(OUT_DIR, "cat2id.json"), "r", encoding="utf-8") as f:
    cat2id = json.load(f)
id2cat = {v:k for k,v in cat2id.items()}

centroids = {
    "x": torch.load(os.path.join(OUT_DIR, "centroids_x.pt")).cpu().numpy(),
    "y": torch.load(os.path.join(OUT_DIR, "centroids_y.pt")).cpu().numpy(),
    "w": torch.load(os.path.join(OUT_DIR, "centroids_w.pt")).cpu().numpy(),
    "h": torch.load(os.path.join(OUT_DIR, "centroids_h.pt")).cpu().numpy(),
}

tokens_train = torch.load(os.path.join(OUT_DIR, "tokens_train.pt")).cpu()  # [N,M,5]
pad_c = vocab_meta["c"]["pad_id"]
pad_b = vocab_meta["x"]["pad_id"]

# -----------------------
# Build train id list (same split as builder)
# If you saved split ids, use them. If not, we reconstruct by re-loading semantic json list
# and using the same split function approach: easiest is to re-list semantic files and re-split.
# -----------------------
def split_ids(n: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return train_idx, val_idx, test_idx

# list semantic json files in the exact same order used by builder (sorted)
json_files = sorted([f for f in os.listdir(SEM_DIR) if f.lower().endswith(".json")])
train_idx, _, _ = split_ids(
    n=len(json_files),
    seed=vocab_meta.get("seed", 42),
    train_ratio=vocab_meta["split"]["train"],
    val_ratio=vocab_meta["split"]["val"],
    test_ratio=vocab_meta["split"]["test"]
)
# map train row index -> screen_id
train_ids = [os.path.splitext(json_files[i])[0] for i in train_idx]

# -----------------------
# Helpers
# -----------------------
def find_image_for_id(screen_id: str) -> str:
    """
    Tries common filename patterns. Returns path or raises FileNotFoundError.
    """
    patterns = [
        f"{screen_id}.png",
        f"{screen_id}.jpg",
        f"screen_{screen_id}.png",
        f"screen_{screen_id}.jpg",
    ]
    for d in IMG_DIR_CANDIDATES:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            path = os.path.join(d, p)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"No image found for id={screen_id} in candidates")

def decode_row_to_boxes(tokens_row: torch.Tensor):
    """
    tokens_row: [M,5] -> list of dicts with cat_id + normalized box center/size
    """
    boxes = []
    for t in tokens_row.tolist():
        c_id, x_id, y_id, w_id, h_id = t
        if c_id == pad_c:
            continue
        if x_id == pad_b or y_id == pad_b or w_id == pad_b or h_id == pad_b:
            continue
        x = float(centroids["x"][x_id])
        y = float(centroids["y"][y_id])
        w = float(centroids["w"][w_id])
        h = float(centroids["h"][h_id])
        boxes.append({"c_id": int(c_id), "x": x, "y": y, "w": w, "h": h})
    return boxes

def draw_boxes_on_image(img: Image.Image, boxes, draw_labels=False):
    """
    Draws rectangles from center (x,y) and size (w,h) normalized.
    """
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # Optional font
    font = None
    if draw_labels:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    for b in boxes[:MAX_BOXES]:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        x0 = (x - w/2) * W
        y0 = (y - h/2) * H
        x1 = (x + w/2) * W
        y1 = (y + h/2) * H

        # clamp to image bounds
        x0 = max(0, min(W-1, x0))
        y0 = max(0, min(H-1, y0))
        x1 = max(0, min(W-1, x1))
        y1 = max(0, min(H-1, y1))

        # draw rectangle (default stroke color)
        draw.rectangle([x0, y0, x1, y1], width=2)

        if draw_labels:
            c_id = b["c_id"]
            label = f"{c_id}:{id2cat.get(c_id,'?')}"
            draw.text((x0+2, y0+2), label, font=font)

    return img

def make_grid(images, cols=6, pad=8, bg=(255,255,255)):
    """
    images: list of PIL images (can be different sizes). We'll resize to a common thumbnail.
    """
    if not images:
        return None
    # thumbnail size
    thumb_w = 360
    thumb_h = 640
    thumbs = []
    for im in images:
        im2 = im.copy()
        im2.thumbnail((thumb_w, thumb_h))
        # paste centered into fixed frame
        canvas = Image.new("RGB", (thumb_w, thumb_h), bg)
        x = (thumb_w - im2.size[0])//2
        y = (thumb_h - im2.size[1])//2
        canvas.paste(im2, (x,y))
        thumbs.append(canvas)

    rows = int(np.ceil(len(thumbs)/cols))
    grid_w = cols*thumb_w + (cols+1)*pad
    grid_h = rows*thumb_h + (rows+1)*pad
    grid = Image.new("RGB", (grid_w, grid_h), bg)

    for i, im in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = pad + c*(thumb_w+pad)
        y = pad + r*(thumb_h+pad)
        grid.paste(im, (x,y))
    return grid

# -----------------------
# Render K samples
# -----------------------
random.seed(SEED)
choices = random.sample(range(len(train_ids)), k=min(K, len(train_ids)))

rendered = []
skipped = 0

for j, ti in enumerate(choices):
    screen_id = train_ids[ti]
    # tokens row aligned to train split order
    tok_row = tokens_train[ti]  # [M,5]
    boxes = decode_row_to_boxes(tok_row)

    try:
        img_path = find_image_for_id(screen_id)
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("[skip]", screen_id, "->", e)
        skipped += 1
        continue

    overlay = img.copy()
    overlay = draw_boxes_on_image(overlay, boxes, draw_labels=DRAW_LABELS)

    out_path = os.path.join(OVERLAY_OUT, f"{screen_id}_overlay.png")
    overlay.save(out_path)
    rendered.append(overlay)

print("\nDone. Rendered:", len(rendered), "Skipped:", skipped)
print("Saved overlays to:", OVERLAY_OUT)

grid = make_grid(rendered, cols=6)
if grid is not None:
    grid_path = os.path.join(OVERLAY_OUT, "grid_overlays.png")
    grid.save(grid_path)
    print("Saved grid:", grid_path)

# If you're in Colab, show the grid
grid

import os, json, random
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
OUT_DIR = "/content/layoutdm_rico_tokens"
SEM_DIR = "/content/semantic_annotations/semantic_annotations"

IMG_DIR_CANDIDATES = [
    "/content/rico_dataset/combined",
]

DEBUG_SCREEN_ID = "2"  # <-- CAMBIA ESTE
SEED = 42  # debe coincidir con vocab_meta["seed"]


# =========================
# Load artifacts
# =========================
tokens_train = torch.load(os.path.join(OUT_DIR, "tokens_train.pt")).cpu()
centroids = {
    "x": torch.load(os.path.join(OUT_DIR, "centroids_x.pt")).cpu().numpy(),
    "y": torch.load(os.path.join(OUT_DIR, "centroids_y.pt")).cpu().numpy(),
    "w": torch.load(os.path.join(OUT_DIR, "centroids_w.pt")).cpu().numpy(),
    "h": torch.load(os.path.join(OUT_DIR, "centroids_h.pt")).cpu().numpy(),
}
with open(os.path.join(OUT_DIR, "vocab_meta.json"), "r", encoding="utf-8") as f:
    vocab_meta = json.load(f)

pad_c = vocab_meta["c"]["pad_id"]
pad_b = vocab_meta["x"]["pad_id"]

TRAIN_RATIO = vocab_meta["split"]["train"]
VAL_RATIO = vocab_meta["split"]["val"]
TEST_RATIO = vocab_meta["split"]["test"]
SEED = vocab_meta.get("seed", SEED)

# =========================
# Helpers (match builder logic)
# =========================
def _walk_nodes(node):
    out = [node]
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            out.extend(_walk_nodes(child))
    return out

def _normalize_bounds(b):
    if not b or len(b) != 4:
        return None
    try:
        x0, y0, a, c = map(float, b)
    except Exception:
        return None

    # assume [x0,y0,x1,y1]
    x1, y1 = a, c
    w = x1 - x0
    h = y1 - y0

    # if degenerate, try [x,y,w,h]
    if w <= 0 or h <= 0:
        w2, h2 = a, c
        if w2 > 0 and h2 > 0:
            x1 = x0 + w2
            y1 = y0 + h2
            w = x1 - x0
            h = y1 - y0

    if w <= 0 or h <= 0:
        return None
    return x0, y0, x1, y1

def can_infer_screen_size(data):
    # Try root bounds first
    rb = data.get("bounds")
    if isinstance(rb, list):
        nb = _normalize_bounds(rb)
        if nb is not None:
            sx0, sy0, sx1, sy1 = nb
            if (sx1 - sx0) > 0 and (sy1 - sy0) > 0:
                return True

    # Else infer from all nodes
    nodes = _walk_nodes(data)
    mins = [float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf")]
    found = False
    for n in nodes:
        b = n.get("bounds")
        if not isinstance(b, list):
            continue
        nb = _normalize_bounds(b)
        if nb is None:
            continue
        x0, y0, x1, y1 = nb
        mins[0] = min(mins[0], x0)
        mins[1] = min(mins[1], y0)
        maxs[0] = max(maxs[0], x1)
        maxs[1] = max(maxs[1], y1)
        found = True

    if not found:
        return False

    return (maxs[0] - mins[0]) > 0 and (maxs[1] - mins[1]) > 0

def split_ids(n, seed, train_ratio, val_ratio, test_ratio):
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return train_idx, val_idx, test_idx

def find_image(screen_id):
    patterns = [f"{screen_id}.png", f"screen_{screen_id}.png", f"{screen_id}.jpg", f"screen_{screen_id}.jpg"]
    for d in IMG_DIR_CANDIDATES:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            path = os.path.join(d, p)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"No image for id={screen_id} in {IMG_DIR_CANDIDATES}")

def decode_row_to_boxes(tokens_row):
    boxes = []
    for t in tokens_row.tolist():
        c_id, x_id, y_id, w_id, h_id = t
        if c_id == pad_c:
            continue
        if x_id == pad_b or y_id == pad_b or w_id == pad_b or h_id == pad_b:
            continue
        boxes.append((
            float(centroids["x"][x_id]),
            float(centroids["y"][y_id]),
            float(centroids["w"][w_id]),
            float(centroids["h"][h_id]),
        ))
    return boxes

# =========================
# 1) Build good_ids (the same base list used by builder)
# =========================
json_files = sorted([f for f in os.listdir(SEM_DIR) if f.lower().endswith(".json")])

good_ids = []
bad_ids = []
for f in json_files:
    p = os.path.join(SEM_DIR, f)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if can_infer_screen_size(data):
            good_ids.append(os.path.splitext(f)[0])
        else:
            bad_ids.append(os.path.splitext(f)[0])
    except Exception:
        bad_ids.append(os.path.splitext(f)[0])

print("Total json:", len(json_files))
print("Good (parseable):", len(good_ids))
print("Bad (unusable):", len(bad_ids))
print("tokens_train rows:", tokens_train.shape[0])

# Sanity: tokens_train debe coincidir con n_train de good_ids split
train_idx, val_idx, test_idx = split_ids(len(good_ids), SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print("Expected train size from good_ids split:", len(train_idx))

# =========================
# 2) Locate DEBUG_SCREEN_ID in train split (if present)
# =========================
if DEBUG_SCREEN_ID not in good_ids:
    print(f"❌ {DEBUG_SCREEN_ID} NO está en good_ids (su JSON no tiene bounds utilizables).")
    # Sugerencia: elige otro id que sí esté en good_ids
    raise SystemExit

pos_in_good = good_ids.index(DEBUG_SCREEN_ID)
if pos_in_good not in train_idx:
    print(f"⚠ {DEBUG_SCREEN_ID} no está en TRAIN. Está en VAL/TEST.")
    # Para debug igual lo podemos buscar en val/test si cargaste tokens_val/test.
    raise SystemExit

train_pos = train_idx.index(pos_in_good)
print("✅ train_pos:", train_pos, "within tokens_train:", tokens_train.shape[0])

tokens_row = tokens_train[train_pos]  # [M,5]

# =========================
# 3) Load original JSON + compute real boxes (normalized)
# =========================
with open(os.path.join(SEM_DIR, f"{DEBUG_SCREEN_ID}.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

# screen size from root (simple) — si root fuera raro, podrías inferir como en builder
rb = _normalize_bounds(data.get("bounds", []))
if rb is None:
    raise RuntimeError("Root bounds invalid for debug; pick another id or infer bbox.")

sx0, sy0, sx1, sy1 = rb
screen_w = (sx1 - sx0)
screen_h = (sy1 - sy0)

nodes = _walk_nodes(data)
nodes = nodes[1:]  # drop root
real_boxes = []
for n in nodes:
    b = n.get("bounds")
    if not isinstance(b, list):
        continue
    nb = _normalize_bounds(b)
    if nb is None:
        continue
    x0, y0, x1, y1 = nb
    w_px = x1 - x0
    h_px = y1 - y0
    if w_px <= 0 or h_px <= 0:
        continue
    x = ((x0 + x1) / 2.0) / screen_w
    y = ((y0 + y1) / 2.0) / screen_h
    w = w_px / screen_w
    h = h_px / screen_h
    real_boxes.append((x,y,w,h))

decoded_boxes = decode_row_to_boxes(tokens_row)

print("Total real boxes:", len(real_boxes))
print("Total decoded boxes:", len(decoded_boxes))

# =========================
# 4) Draw overlay
# =========================
img_path = find_image(DEBUG_SCREEN_ID)
img = Image.open(img_path).convert("RGB")
W, H = img.size

overlay = img.copy()
draw = ImageDraw.Draw(overlay)

# 🔴 real boxes (first M)
for x,y,w,h in real_boxes[:tokens_train.shape[1]]:
    x0 = (x - w/2) * W
    y0 = (y - h/2) * H
    x1 = (x + w/2) * W
    y1 = (y + h/2) * H
    draw.rectangle([x0,y0,x1,y1], outline="red", width=2)

# 🟢 decoded boxes (all non-pad)
for x,y,w,h in decoded_boxes:
    x0 = (x - w/2) * W
    y0 = (y - h/2) * H
    x1 = (x + w/2) * W
    y1 = (y + h/2) * H
    draw.rectangle([x0,y0,x1,y1], outline="green", width=2)

plt.figure(figsize=(6, 12))
plt.imshow(overlay)
plt.axis("off")
plt.title(f"ID={DEBUG_SCREEN_ID}  red=real  green=decoded")
plt.show()

import os, json
from PIL import Image

SEM_DIR = "/content/semantic_annotations/semantic_annotations"
IMG_PATH = "/content/rico_dataset/combined/2.jpg"   # AJUSTA
DEBUG_SCREEN_ID = "2"

with open(os.path.join(SEM_DIR, f"{DEBUG_SCREEN_ID}.json"), "r") as f:
    data = json.load(f)

img = Image.open(IMG_PATH).convert("RGB")
print("Image size:", img.size)

rb = data.get("bounds")
print("Root bounds raw:", rb)

# si root es [x0,y0,x1,y1]
sx0, sy0, sx1, sy1 = rb
screen_w = sx1 - sx0
screen_h = sy1 - sy0
print("Computed screen_w/h:", screen_w, screen_h)
print("Root offset sx0,sy0:", sx0, sy0)

print("Image aspect:", img.size[0]/img.size[1])
print("Root aspect :", screen_w/screen_h)

import os, json, math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

SEM_DIR = "/content/semantic_annotations/semantic_annotations"
DEBUG_SCREEN_ID = "2"
IMG_PATH = f"/content/rico_dataset/combined/{DEBUG_SCREEN_ID}.jpg"

# ---------- helpers ----------
def walk(node):
    out = [node]
    for c in node.get("children", []) or []:
        if isinstance(c, dict):
            out.extend(walk(c))
    return out

def normalize_bounds(b):
    # assume [x0,y0,x1,y1]
    if not b or len(b) != 4: return None
    x0,y0,x1,y1 = map(float, b)
    if (x1-x0) <= 0 or (y1-y0) <= 0:
        return None
    return x0,y0,x1,y1

def infer_base_resolution(root_bounds, img_w, img_h):
    """
    Heurística robusta:
    - si x0+x1 parece un ancho típico (720/1080/1440), úsalo.
    - si no, intenta encajar con candidatos comunes para que el scale sea consistente en W y H.
    """
    x0,y0,x1,y1 = root_bounds
    sum_w = x0 + x1
    sum_h = y0 + y1

    Wcands = [720, 1080, 1440, 1536]
    Hcands = [1280, 1920, 2160, 2560, 2960]

    # 1) intentar con sum_w cercano a un candidato
    W0 = min(Wcands, key=lambda W: abs(W - sum_w))
    # scale por ancho
    s = img_w / W0

    # 2) elegir H0 que mejor cuadre con ese scale
    H0 = min(Hcands, key=lambda H: abs(img_h - H*s))

    return int(W0), int(H0), float(s)

# ---------- load ----------
with open(os.path.join(SEM_DIR, f"{DEBUG_SCREEN_ID}.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

img = Image.open(IMG_PATH).convert("RGB")
W_img, H_img = img.size

rb = normalize_bounds(data.get("bounds"))
if rb is None:
    raise RuntimeError("Root bounds inválido para este debug.")

sx0, sy0, sx1, sy1 = rb
root_w = sx1 - sx0
root_h = sy1 - sy0

# Inferir resolución base del JSON vs la imagen
base_w, base_h, scale = infer_base_resolution(rb, W_img, H_img)

print("Image:", (W_img, H_img))
print("Root bounds:", rb, "root_wh:", (root_w, root_h))
print("Inferred base:", (base_w, base_h), "scale:", scale)

nodes = walk(data)[1:]  # drop root
abs_boxes = []
for n in nodes:
    b = normalize_bounds(n.get("bounds"))
    if b is None:
        continue
    abs_boxes.append(b)

print("Total boxes:", len(abs_boxes))

# ---------- draw ----------
overlay = img.copy()
draw = ImageDraw.Draw(overlay)

# AMARILLO: ABS(JSON) -> IMG (scale)
for x0,y0,x1,y1 in abs_boxes:
    X0 = x0 * scale
    Y0 = y0 * scale
    X1 = x1 * scale
    Y1 = y1 * scale
    draw.rectangle([X0,Y0,X1,Y1], outline="yellow", width=2)

# ROJO: tu normalización vieja (sin offset) -> IMG
for x0,y0,x1,y1 in abs_boxes:
    x_c = (x0+x1)/2
    y_c = (y0+y1)/2
    w = (x1-x0)
    h = (y1-y0)
    # ❌ mal: no resta offset y normaliza por root_w/root_h
    xn = (x_c) / root_w
    yn = (y_c) / root_h
    wn = w / root_w
    hn = h / root_h

    X0 = (xn - wn/2) * W_img
    Y0 = (yn - hn/2) * H_img
    X1 = (xn + wn/2) * W_img
    Y1 = (yn + hn/2) * H_img
    draw.rectangle([X0,Y0,X1,Y1], outline="red", width=2)

# AZUL: coords relativas al bbox del root (con offset) -> IMG
for x0,y0,x1,y1 in abs_boxes:
    x_c = (x0+x1)/2
    y_c = (y0+y1)/2
    w = (x1-x0)
    h = (y1-y0)
    # ✅ relativo a contenido (root bbox)
    xn = (x_c - sx0) / root_w
    yn = (y_c - sy0) / root_h
    wn = w / root_w
    hn = h / root_h

    X0 = (xn - wn/2) * W_img
    Y0 = (yn - hn/2) * H_img
    X1 = (xn + wn/2) * W_img
    Y1 = (yn + hn/2) * H_img
    draw.rectangle([X0,Y0,X1,Y1], outline="blue", width=2)

plt.figure(figsize=(6,12))
plt.imshow(overlay)
plt.axis("off")
plt.title("yellow=ABS->scaled (correct if image is resized) | red=old | blue=offset-to-root")
plt.show()


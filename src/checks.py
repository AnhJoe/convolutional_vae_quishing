"""Checks that a model notebook runs before it trains anything.

Each function returns a `Checks` object. Call `.report()` on it in a notebook cell: it shows a table of named checks
and raises an error if any of them failed, so a notebook never trains on data that does not match what it assumes.

    data_checks(all_items, summary, scope)     the items table is the one 2_data_processing builds (counts, splits, labels,
                                               URL columns, and for images the measured sizes)
    image_input_checks(raw_dir, all_items)     load_qr_grids gives the arrays the model expects, and cached_grids agrees
    url_input_checks(items)                    encode_urls gives the ids the model expects (padding, cut-off, unknown token)
    evaluation_checks(items, scope)            made-up scores go through save_and_evaluate and give the metrics they must

What the checks cover and what they do not: they test the shared pipeline (files, splits, input arrays, metrics), never a
model. They read a few hundred images or a few thousand URLs, not the whole data set, so they run in seconds.

Typical use in a model notebook:

    all_items = load_items(PROC_DIR)
    summary = json.loads((PROC_DIR / "items_summary.json").read_text())
    checks.data_checks(all_items, summary, SCOPE).report()
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import save_and_evaluate
from src.loaders import CANVAS, MAX_URL_LEN, VOCAB_SIZE, cached_grids, decode_url, encode_urls, load_qr_grid, load_qr_grids
from src.prepare_data import data_version
from src.splits import SPLIT_NAMES

URL_FLOOR_FEATURES = ["url_len", "has_scheme", "host_is_ip", "n_host_labels", "path_len", "digit_ratio"]
IMAGE_FLOOR_FEATURES = ["modules", "dark_frac"]
IMAGE_COLUMNS = ["side", "modules", "border", "dark_frac", "qr_version"]
MODULE_PX = 6


# ----------------------------------------------------------------------------------------------------------------------
# A list of named checks
# ----------------------------------------------------------------------------------------------------------------------
class Checks:
    """A list of named pass/fail checks (plus information rows). `report()` shows the table and raises if any failed."""

    def __init__(self, title: str):
        self.title = title
        self.rows: list[dict] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.rows.append({"check": name, "result": "pass" if bool(passed) else "FAIL", "detail": str(detail)})

    def info(self, name: str, detail: str) -> None:
        self.rows.append({"check": name, "result": "info", "detail": str(detail)})

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "result", "detail"])

    def report(self) -> None:
        t = self.table()
        _show(t, self.title)
        failed = t.loc[t["result"] == "FAIL", "check"].tolist()
        if failed:
            raise AssertionError(f"{len(failed)} check(s) failed: " + "; ".join(failed))


def _show(t: pd.DataFrame, title: str) -> None:
    """Display the table in a notebook (wrapped text, failures in red). Falls back to plain text elsewhere."""
    try:
        from IPython.display import display
        styled = (t.style.hide(axis="index").set_caption(title)
                  .set_table_styles([
                      {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"), ("font-weight", "bold"), ("padding", "4px 0 8px 0")]},
                      {"selector": "th, td", "props": [("white-space", "normal"), ("overflow-wrap", "anywhere"), ("max-width", "520px"), ("text-align", "left"), ("vertical-align", "top")]},
                  ])
                  .map(lambda v: "color: #b00020; font-weight: bold" if v == "FAIL" else "", subset=["result"]))
        display(styled)
    except Exception:
        print(title)
        print(t.to_string(index=False))


def _refused(fn) -> bool:
    """True if calling fn raises ValueError (the evaluation functions refuse malformed scores with ValueError)."""
    try:
        fn()
    except ValueError:
        return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
# The items table
# ----------------------------------------------------------------------------------------------------------------------
def data_checks(all_items: pd.DataFrame, summary: dict, scope: str) -> Checks:
    """Check the whole items table (load_items with no filter) against items_summary.json and against itself.

    scope is "all" for URL notebooks and "canvas" for image notebooks. The canvas scope adds checks on the image columns.
    """
    c = Checks(f"Data checks (scope: {scope})")
    t = all_items
    c.add("row count matches items_summary.json", len(t) == summary["rows"], f"{len(t):,} rows")
    version = data_version(t)
    c.add("data version matches items_summary.json", version == summary["data_version"],
          f"{version}. Everyone on the team should see the same value")
    c.add("item_id equals the row position", bool((t["item_id"].to_numpy() == np.arange(len(t))).all()))
    c.add("label matches the class folder", bool((t["label"] == (t["cls"] == "malicious").astype(int)).all()))

    counts = {s: {k: int(v) for k, v in g["cls"].value_counts().items()} for s, g in t.groupby("split")}
    c.add("rows per split and class match items_summary.json", counts == summary["rows_by_split_and_class"],
          "; ".join(f"{s}: {counts[s].get('benign', 0):,} benign, {counts[s].get('malicious', 0):,} malicious" for s in SPLIT_NAMES))
    c.add("every split has benign and malicious rows", all(counts.get(s, {}).get(k, 0) > 0 for s in SPLIT_NAMES for k in ("benign", "malicious")))
    split_domains = t.groupby(["cls", "domain"])["split"].nunique()
    c.add("no domain appears in two splits of the same class", int(split_domains.max()) == 1)
    share = t.groupby("split")["label"].mean()
    overall = t["label"].mean()
    c.add("malicious share of every split is within 1 point of the overall share", bool((share - overall).abs().max() < 0.01),
          ", ".join(f"{s} {share[s]:.3f}" for s in SPLIT_NAMES) + f" (overall {overall:.3f})")
    c.add("no empty URL, and url_len equals the URL length", bool((t["url"].str.len() > 0).all() and (t["url_len"] == t["url"].str.len()).all()))
    c.add(f"truncated is true exactly for URLs longer than {MAX_URL_LEN}", bool((t["truncated"] == (t["url_len"] > MAX_URL_LEN)).all()),
          f"{int(t['truncated'].sum()):,} truncated")

    if scope == "all":
        missing = int(t[URL_FLOOR_FEATURES].isna().sum().sum())
        c.add("URL floor features have no missing values", missing == 0, ", ".join(URL_FLOOR_FEATURES))
    elif scope == "canvas":
        inside = t[t["fits_canvas"]]
        outside = {k: int(v) for k, v in t.loc[~t["fits_canvas"]].groupby("cls").size().items()}
        c.add("codes outside the canvas match items_summary.json", outside == summary["outside_canvas_by_class"],
              f"{outside.get('benign', 0):,} benign and {outside.get('malicious', 0):,} malicious are left out of the image models")
        c.add(f"every code inside the canvas has a grid of at most {CANVAS}", bool(inside["grid"].le(CANVAS).all()), f"largest grid {inside['grid'].max():.0f}")
        c.add("image measurements are filled for every code inside the canvas", int(inside[IMAGE_COLUMNS].isna().sum().sum()) == 0, ", ".join(IMAGE_COLUMNS))
        c.add("image floor features have no missing values", int(inside[IMAGE_FLOOR_FEATURES].isna().sum().sum()) == 0, ", ".join(IMAGE_FLOOR_FEATURES))
        c.add(f"PNG width / {MODULE_PX} equals the grid size for every code inside the canvas", bool((inside["side"] / MODULE_PX == inside["grid"]).all()),
              "the file header and the pixel measurement agree")
        c.add("modules + 2 x border equals the grid size for every code inside the canvas", bool(np.isclose(inside["modules"] + 2 * inside["border"], inside["grid"]).all()),
              "the module count and quiet zone agree with the image width")
        micro = inside.groupby("cls")["qr_version"].apply(lambda s: (s == 0).mean())
        c.info("Micro QR share inside the canvas", f"benign {micro.get('benign', float('nan')):.3f}, malicious {micro.get('malicious', float('nan')):.3f} (qr_version 0)")
    else:
        raise ValueError("scope must be 'all' or 'canvas'")
    return c


# ----------------------------------------------------------------------------------------------------------------------
# Model input
# ----------------------------------------------------------------------------------------------------------------------
def image_input_checks(raw_dir: Path | str, all_items: pd.DataFrame, n_per_class: int = 100, seed: int = 0) -> Checks:
    """Read a few hundred training images with load_qr_grids and compare the arrays with the items table.

    The sample is n_per_class benign and n_per_class malicious training codes plus the three smallest and three largest
    grids of the whole table. all_items must be the full table, because one code outside the canvas is used to check that
    it is refused.
    """
    c = Checks("Image input checks")
    inside = all_items[all_items["fits_canvas"]]
    train = inside[inside["split"] == "train"]
    sample = pd.concat([train[train["cls"] == k].sample(n_per_class, random_state=seed) for k in ("benign", "malicious")]
                       + [inside.nsmallest(3, "grid"), inside.nlargest(3, "grid")]).drop_duplicates("item_id")
    start = time.time()
    g = load_qr_grids(raw_dir, sample)
    seconds = time.time() - start
    grid = sample["grid"].to_numpy().astype(int)
    border = sample["border"].to_numpy().astype(int)
    idx = range(len(sample))

    c.add(f"array shape is (n, {CANVAS}, {CANVAS}) and the type is uint8", g.shape == (len(sample), CANVAS, CANVAS) and g.dtype == np.uint8, f"{g.shape}, {g.dtype}")
    c.add("values are only 0 (light module) and 1 (dark module)", set(np.unique(g).tolist()) <= {0, 1})
    c.add("nothing is drawn outside the code's own grid (top-left anchoring)",
          all(g[i, grid[i]:, :].sum() == 0 and g[i, :, grid[i]:].sum() == 0 for i in idx), "including the smallest and largest grids")
    c.add("the quiet zone is light on all four sides and the first dark module sits at (border, border)",
          all(g[i, :border[i], :].sum() == 0 and g[i, :, :border[i]].sum() == 0
              and g[i, grid[i] - border[i]:grid[i], :].sum() == 0 and g[i, :, grid[i] - border[i]:grid[i]].sum() == 0
              and g[i, border[i], border[i]] == 1 for i in idx))
    gap = float(np.abs(g.sum(axis=(1, 2)) / grid ** 2 - sample["dark_frac"].to_numpy()).max())
    c.add("the dark share of each array equals dark_frac of the items table", gap < 1e-9, f"largest difference {gap:.1e}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grids.npz"
        first = cached_grids(raw_dir, sample.head(40), path)
        again = cached_grids(raw_dir, sample.head(40), path)
        other = cached_grids(raw_dir, sample.iloc[10:30], path)
    c.add("cached_grids returns the same arrays as load_qr_grids, and rebuilds when the items change",
          np.array_equal(first, g[:40]) and np.array_equal(again, first) and np.array_equal(other, g[10:30]))

    too_big = all_items[~all_items["fits_canvas"]].head(1)
    if len(too_big):
        r = too_big.iloc[0]
        c.add("a code larger than the canvas is refused instead of cropped", _refused(lambda: load_qr_grid(raw_dir, r["cls"], r["file"])), f"grid {r['grid']:.0f}")
    n_train = int((train["cls"] == "benign").sum())
    c.info("reading speed", f"{1000 * seconds / len(sample):.1f} ms per image in this run, so about {n_train * seconds / len(sample) / 60:.0f} minutes "
                            f"for the {n_train:,} benign train codes. Use cached_grids so the images are read once")
    return c


def url_input_checks(items: pd.DataFrame, n: int = 3000, seed: int = 0) -> Checks:
    """Encode a few thousand URLs with encode_urls and check the ids against the URLs they came from.

    Also checks URLs with characters outside printable ASCII (the unknown token) and URLs longer than the maximum length.
    """
    c = Checks("URL input checks")
    sample = items.sample(n, random_state=seed)
    urls = sample["url"].tolist()
    ids = encode_urls(urls)
    shown = lambda u: "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in u[:MAX_URL_LEN])     # what decode_url must return

    c.add(f"array shape is (n, {MAX_URL_LEN}) and the type is int16", ids.shape == (n, MAX_URL_LEN) and ids.dtype == np.int16, f"{ids.shape}, {ids.dtype}")
    c.add(f"ids are between 0 and {VOCAB_SIZE - 1} (vocabulary size {VOCAB_SIZE})", int(ids.min()) >= 0 and int(ids.max()) < VOCAB_SIZE, f"smallest {ids.min()}, largest {ids.max()}")
    nonpad = ids != 0
    c.add("padding (id 0) only comes after the last character", bool((np.diff(nonpad.astype(int), axis=1) <= 0).all()))
    c.add(f"each row has min(URL length, {MAX_URL_LEN}) real ids", bool((nonpad.sum(axis=1) == np.minimum(sample["url_len"].to_numpy(), MAX_URL_LEN)).all()))
    c.add("decode_url(encode_urls(url)) gives back the URL (cut at the maximum length, unknown characters as ?)",
          all(decode_url(row) == shown(u) for row, u in zip(ids, urls)))

    odd = items.loc[~items["url"].str.fullmatch(r"[ -~]*")].head(20)
    odd_ids = encode_urls(odd["url"])
    c.add("a URL with a character outside printable ASCII gets the unknown token (id 1)", len(odd) > 0 and bool(((odd_ids == 1).sum(axis=1) > 0).all()), f"{len(odd)} URLs tried")
    long = items.loc[items["truncated"]].head(20)
    long_ids = encode_urls(long["url"])
    c.add(f"a URL longer than {MAX_URL_LEN} is cut off at {MAX_URL_LEN}", len(long) > 0 and bool(((long_ids != 0).sum(axis=1) == MAX_URL_LEN).all())
          and all(decode_url(row) == shown(u) for row, u in zip(long_ids, long["url"])), f"{len(long)} URLs tried")
    c.info("unknown token share", f"{100 * (ids == 1).sum() / nonpad.sum():.4f}% of the real ids in this sample")
    return c


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------
def evaluation_checks(items: pd.DataFrame, scope: str, seed: int = 0) -> Checks:
    """Run made-up scores through save_and_evaluate in a temporary folder and compare with the metrics they must give.

    Random scores must give AUROC near 0.5 and AUPRC near the malicious share. Perfect scores (the label itself) must give
    1 everywhere, and reversed scores must give AUROC 0. Malformed score files must be refused.
    """
    c = Checks(f"Evaluation checks (scope: {scope}, made-up scores in a temporary folder)")
    split = {s: items[items["split"] == s] for s in ("validation", "test")}
    if scope == "canvas":
        split = {s: t[t["fits_canvas"]] for s, t in split.items()}
    val, test = split["validation"], split["test"]
    rng = np.random.default_rng(seed)

    def run(val_scores, test_scores, val_ids=None, test_ids=None):
        with tempfile.TemporaryDirectory() as tmp:
            return save_and_evaluate(tmp, items, scope, "check", "made_up", seed,
                                     validation=(val["item_id"] if val_ids is None else val_ids, val_scores),
                                     test=(test["item_id"] if test_ids is None else test_ids, test_scores)).iloc[0]

    r = run(rng.normal(size=len(val)), rng.normal(size=len(test)))
    c.add("random scores give AUROC near 0.5", abs(r["auroc"] - 0.5) < 0.01, f"AUROC {r['auroc']:.4f}")
    c.add("random scores give AUPRC near the malicious share", abs(r["auprc"] - r["prevalence"]) < 0.01, f"AUPRC {r['auprc']:.4f}, malicious share {r['prevalence']:.4f}")
    c.add("random scores flag about 1% and 5% of test benign items at the validation thresholds",
          abs(r["test_fpr_at_1pct"] - 0.01) < 0.004 and abs(r["test_fpr_at_5pct"] - 0.05) < 0.01,
          f"{100 * r['test_fpr_at_1pct']:.2f}% and {100 * r['test_fpr_at_5pct']:.2f}%")

    p = run(val["label"].to_numpy(float), test["label"].to_numpy(float))
    c.add("perfect scores give AUROC 1, AUPRC 1 and recall 1 with no false alarms",
          p["auroc"] == 1 and p["auprc"] == 1 and p["recall_at_1pct_fpr"] == 1 and p["recall_at_5pct_fpr"] == 1 and p["test_fpr_at_1pct"] == 0,
          f"AUROC {p['auroc']:.3f}, AUPRC {p['auprc']:.3f}, recall at 1% FPR {p['recall_at_1pct_fpr']:.3f}")
    q = run(-val["label"].to_numpy(float), -test["label"].to_numpy(float))
    c.add("reversed scores give AUROC 0 (higher must mean more malicious)", q["auroc"] == 0, f"AUROC {q['auroc']:.3f}")

    scores = rng.normal(size=len(val))
    t_scores = rng.normal(size=len(test))
    c.add("a validation file that misses one item is refused", _refused(lambda: run(scores[1:], t_scores, val_ids=val["item_id"].to_numpy()[1:])))
    c.add("a validation file with an item that is not in the split is refused",
          _refused(lambda: run(np.append(scores, 0.0), t_scores, val_ids=np.append(val["item_id"].to_numpy(), -1))))
    c.add("scores with a missing value (NaN) are refused", _refused(lambda: run(np.where(np.arange(len(val)) == 0, np.nan, scores), t_scores)))
    return c

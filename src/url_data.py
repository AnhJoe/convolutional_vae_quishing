"""Offline ingestion, lexical features, and shared domain partitions."""
import hashlib
import ipaddress
import re
from functools import lru_cache
from urllib.parse import urlsplit

import pandas as pd
import tldextract

from .splits import SPLIT_NAMES, joint_class_split

FEATURES = ["length", "has_scheme", "ip_host", "host_labels", "path_length", "digit_ratio"]
SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*)://", re.I | re.ASCII)
EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True,
)
PAD, UNK, BOS = range(3)
VOCAB_SIZE = 98
MAX_CHARS = 256


@lru_cache(maxsize=65536)
def host_group(host):
    if not host or not re.fullmatch(r"[a-z0-9._:%-]+", host):
        return "invalid-host", False
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        parts = EXTRACT(host)
        # Keep subdomains together even when the suffix is absent from the PSL.
        fallback = ".".join(host.split(".")[-2:]) if not parts.suffix else host
        domain = f"{parts.domain}.{parts.suffix}" if parts.domain and parts.suffix else fallback
        return "dns:" + domain, False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return "ip:" + str(address), True


def url_features(raw):
    value = raw.strip()
    prefix = SCHEME.match(value)
    scheme = prefix[1].lower() if prefix else "none"
    try:
        candidate = value if prefix or value.startswith("//") else "//" + value
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").rstrip(".")
        host = host.encode("idna").decode("ascii").lower()
        path_length = len(parsed.path)
    except (ValueError, UnicodeError):
        host, path_length = "", 0
    domain, is_ip = host_group(host)
    host_labels = len(host.split(".")) if host and ":" not in host else int(bool(host))
    digit_ratio = sum("0" <= char <= "9" for char in raw) / len(raw)
    return domain, scheme, len(raw), bool(prefix), is_ip, host_labels, path_length, digit_ratio


def prepare(benign_csv, malicious_csv, split_seed):
    sources = {}
    frames = []
    for label, path in enumerate((benign_csv, malicious_csv)):
        frame = pd.read_csv(path, usecols=["url"], dtype=str, keep_default_na=False,
                            skip_blank_lines=False)
        frame["label"] = label
        frame["row_id"] = str(label) + ":" + frame.index.astype(str)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        sources[str(label)] = {
            "name": path.name, "sha256": checksum,
            "rows": len(frame),
        }
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    blank = ~frame.url.str.strip().astype(bool)
    frame = frame.loc[~blank].copy()
    conflict = frame.url.isin(set(frame.loc[frame.label == 0, "url"]) &
                              set(frame.loc[frame.label == 1, "url"]))
    frame = frame.loc[~conflict].copy()
    duplicate = frame.duplicated(["label", "url"])
    frame = frame.loc[~duplicate].reset_index(drop=True)
    if frame.label.nunique() != 2:
        raise ValueError("Both classes must have usable URLs.")
    fields = ["domain", "scheme", *FEATURES]
    frame[fields] = pd.DataFrame.from_records(map(url_features, frame.url), columns=fields)
    frame["split"] = joint_class_split(frame, split_seed)
    overlap = int(frame.groupby("domain").split.nunique().gt(1).sum())
    if overlap:
        raise RuntimeError(f"Split leakage: {overlap} domains span partitions.")
    splits = {}
    for name in SPLIT_NAMES:
        part = frame.loc[frame.split == name]
        classes = {}
        for label in (0, 1):
            cohort = part.loc[part.label == label]
            if not len(cohort):
                raise ValueError(f"{name} has no class {label}. Need more distinct domains.")
            classes[str(label)] = {
                "rows": len(cohort), "domains": int(cohort.domain.nunique()),
                "ip_host": int(cohort.ip_host.sum()),
                "truncated": int((cohort.length > MAX_CHARS).sum()),
                "unparseable_host": int((cohort.domain == "invalid-host").sum()),
            }
        splits[name] = {
            "classes": classes, "malicious_prevalence": float(part.label.mean()),
            "domains": int(part.domain.nunique()),
        }
    audit = {
        "input_csvs": sources, "split_seed": split_seed, "splits": splits,
        "blank_rows_removed": int(blank.sum()), "conflicting_rows_removed": int(conflict.sum()),
        "same_class_duplicates_removed": int(duplicate.sum()), "cross_split_domains": overlap,
    }
    return frame, audit


def encode(raw):
    return [ord(char) - 29 if " " <= char <= "~" else UNK for char in raw[:MAX_CHARS]]

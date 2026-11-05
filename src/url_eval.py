"""Validation-only calibration and paired domain bootstrap on the test set."""
import numpy as np

FPRS = (0.01, 0.05)
METRICS = ("auroc", "auprc_ap", "recall_at_1pct", "test_fpr_at_1pct",
           "recall_at_5pct", "test_fpr_at_5pct")
MIN_ROWS = 100
MIN_DOMAINS = 10


def calibrate(scores, benign):
    ordered = np.sort(scores[benign])
    # Empirical CDF inverse with strict > at prediction time. Ties are negative.
    return [float(ordered[int(np.ceil((1 - rate) * len(ordered))) - 1]) for rate in FPRS]


class RankedScores:
    """Presort once per scorer and stratum, then calculate weighted tied-rank metrics."""

    def __init__(self, scores, labels, thresholds):
        self.order = np.argsort(-scores, kind="stable")
        ranked = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
        self.labels = labels[self.order]
        self.cutoffs = [int((ranked > threshold).sum()) for threshold in thresholds]

    def metrics(self, weights):
        weights = weights[self.order]
        positive = weights * self.labels
        pos = np.add.reduceat(positive, self.starts)
        neg = np.add.reduceat(weights - positive, self.starts)
        cumulative_pos = pos.cumsum()
        cumulative_neg = neg.cumsum()
        n_pos, n_neg = cumulative_pos[-1], cumulative_neg[-1]
        if not n_pos or not n_neg:
            return np.full(len(METRICS), np.nan)
        auc = np.dot(pos.astype(float), n_neg - cumulative_neg + 0.5 * neg) / (n_pos * n_neg)
        cumulative = cumulative_pos + cumulative_neg
        precision = np.divide(cumulative_pos, cumulative, where=cumulative > 0,
                              out=np.zeros_like(cumulative, dtype=float))
        ap = np.dot(pos, precision) / n_pos
        rates = []
        for cutoff in self.cutoffs:
            hits = positive[:cutoff].sum()
            rates.extend((hits / n_pos, (weights[:cutoff].sum() - hits) / n_neg))
        return np.array([auc, ap, *rates])


def interval(point, draws):
    values = draws[np.isfinite(draws)]
    limits = np.quantile(values, [0.025, 0.975]).tolist() if len(values) >= 20 else None
    return {"estimate": float(point) if np.isfinite(point) else None,
            "ci95": limits, "valid_bootstraps": len(values)}


def counts(part):
    return {str(label): {"rows": int(len(cohort)), "domains": int(cohort.domain.nunique())}
            for label in (0, 1) for cohort in [part.loc[part.label == label]]}


def eligible(cohort_counts):
    return all(info["rows"] >= MIN_ROWS and info["domains"] >= MIN_DOMAINS
               for info in cohort_counts.values())


def strata(validation, test):
    edges = np.unique(np.quantile(validation.length, np.arange(1, 10) / 10,
                                  method="inverted_cdf"))
    edges = edges[edges < validation.length.max()]
    val_bins = np.searchsorted(edges, validation.length.to_numpy())
    test_bins = np.searchsorted(edges, test.length.to_numpy())
    for index in range(len(edges) + 1):
        limits = {"min_inclusive": None if index == 0 else int(edges[index - 1]) + 1,
                  "max_inclusive": None if index == len(edges) else int(edges[index])}
        yield f"length_{index + 1}", val_bins == index, test_bins == index, limits
    schemes = ("none", "http", "https")
    for scheme in (*schemes, "other"):
        if scheme == "other":
            yield "scheme_other", ~validation.scheme.isin(schemes), ~test.scheme.isin(schemes), {}
        else:
            yield "scheme_" + scheme, validation.scheme == scheme, test.scheme == scheme, {}


def evaluate_cohort(cohort, scores, thresholds, repeats, seed, vae_names, ranking_only=False):
    labels = cohort.label.to_numpy()
    _, domains = np.unique(cohort.domain, return_inverse=True)
    n_domains = int(domains.max()) + 1
    rankings = [RankedScores(values, labels, thresholds[name]) for name, values in scores.items()]
    points = np.stack([ranking.metrics(np.ones(len(cohort), dtype=np.int64)) for ranking in rankings])
    draws = np.empty((repeats, len(rankings), len(METRICS)))
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        sampled = np.bincount(rng.integers(n_domains, size=n_domains), minlength=n_domains)
        weights = sampled[domains]
        draws[repeat] = [ranking.metrics(weights) for ranking in rankings]
    names = list(scores)
    metrics = {
        name: {metric: interval(points[index, col], draws[:, index, col])
               for col, metric in enumerate(METRICS) if not ranking_only or col == 0}
        for index, name in enumerate(names)
    }
    contrasts = {}
    for name in vae_names:
        index = names.index(name)
        contrasts[name] = {}
        for floor in ("floor_a", "floor_b"):
            other = names.index(floor)
            contrasts[name][floor] = interval(points[index, 0] - points[other, 0],
                                              draws[:, index, 0] - draws[:, other, 0])
    return {"metrics": metrics, "paired_auroc_deltas": contrasts}


def evaluate(validation, test, validation_scores, test_scores, seeds, repeats, bootstrap_seed):
    thresholds = {name: calibrate(scores, validation.label.to_numpy() == 0)
                  for name, scores in validation_scores.items()}
    vae_names = [name for name in test_scores if name.startswith("vae_")]
    primary = f"vae_reconstruction_s{seeds[0]}"
    overall = evaluate_cohort(test, test_scores, thresholds, repeats, bootstrap_seed, vae_names)
    by_stratum = {}
    # All models are assessed on the same masks and domain draws within a stratum.
    for index, (name, val_mask, test_mask, limits) in enumerate(strata(validation, test)):
        validation_counts, test_counts = counts(validation.loc[val_mask]), counts(test.loc[test_mask])
        entry = {"limits": limits, "validation": validation_counts, "test": test_counts,
                 "eligible_in_validation": eligible(validation_counts),
                 "sufficient_in_test": eligible(test_counts)}
        if entry["eligible_in_validation"] and entry["sufficient_in_test"]:
            entry.update(evaluate_cohort(
                test.loc[test_mask], {key: values[np.asarray(test_mask)] for key, values in test_scores.items()},
                thresholds, repeats, bootstrap_seed + index + 1, vae_names, ranking_only=True,
            ))
        by_stratum[name] = entry

    aggregate = {}
    for family in ("vae_reconstruction", "vae_negative_elbo", "supervised_gru"):
        family_names = [f"{family}_s{seed}" for seed in seeds]
        aggregate[family] = {}
        for metric in METRICS:
            values = [overall["metrics"][name][metric]["estimate"] for name in family_names]
            aggregate[family][metric] = {"mean": float(np.mean(values)),
                                         "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                         "min": min(values), "max": max(values)}

    def positive_ci(contrast):
        limits = contrast["ci95"]
        return limits is not None and limits[0] > 0

    prespecified = [entry for entry in by_stratum.values() if entry["eligible_in_validation"]]
    wins = sum(positive_ci(entry["paired_auroc_deltas"][primary]["floor_a"])
               for entry in prespecified if "paired_auroc_deltas" in entry)
    floor_a = overall["metrics"]["floor_a"]["auroc"]["estimate"]
    worst_seed_delta = aggregate["vae_reconstruction"]["auroc"]["min"] - floor_a
    reference = overall["metrics"][f"supervised_gru_s{seeds[0]}"]["auroc"]["estimate"]
    primary_auc = overall["metrics"][primary]["auroc"]["estimate"]
    return {
        "test_prevalence": float(test.label.mean()), "counts": counts(test),
        "test_ip_host_share_by_class": {str(label): float(test.loc[test.label == label, "ip_host"].mean())
                                        for label in (0, 1)},
        "thresholds": {name: {"scores_strictly_above": cuts,
                              "validation_fpr": [float((validation_scores[name][validation.label == 0]
                                                        > cut).mean()) for cut in cuts]}
                       for name, cuts in thresholds.items()},
        "bootstrap_repeats": repeats, "bootstrap_seed": bootstrap_seed, "primary": primary,
        "overall": overall, "strata": by_stratum, "across_seeds": aggregate,
        "success_criteria": {
            "overall_paired_ci_above_floor_a": positive_ci(overall["paired_auroc_deltas"][primary]["floor_a"]),
            "eligible_strata": len(prespecified), "strata_with_paired_ci_above_floor_a": wins,
            "majority_of_eligible_strata": 2 * wins > len(prespecified),
            "worst_seed_point_delta_vs_floor_a": worst_seed_delta,
            "five_seed_robustness": len(seeds) >= 5 and worst_seed_delta > 0,
            "fraction_of_floor_a_to_gru_gap_closed": None if reference <= floor_a else
            (primary_auc - floor_a) / (reference - floor_a),
        },
    }

"""Run the URL-only quishing experiment from class-specific QR URL CSVs."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .url_data import FEATURES, MAX_CHARS, prepare
from .url_eval import evaluate
from .url_models import Floors, URLGRU, URLVAE, loader


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_model(kind, config, device):
    if kind == "vae":
        return URLVAE(config["embedding"], config["hidden"], config["latent"]).to(device)
    return URLGRU(config["embedding"], config["hidden"]).to(device)


def batch_statistics(model, batch, device, beta, generator=None, samples=1):
    tokens, lengths, labels = batch
    tokens, labels = tokens.to(device), labels.to(device)
    if isinstance(model, URLVAE):
        nll, kl, characters = model(tokens, lengths, generator, samples)
        return (nll + beta * kl) / characters, nll / characters, (nll + kl) / characters, kl
    logits = model(tokens, lengths)
    return F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none"), logits


@torch.inference_mode()
def score(model, batches, device, samples, keep_scores=False):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(275)
    totals = None
    rows = 0
    collected = []
    for batch in batches:
        values = batch_statistics(model, batch, device, 1.0, generator, samples)
        batch_totals = torch.stack([value.double().sum() for value in values])
        totals = batch_totals if totals is None else totals + batch_totals
        rows += len(batch[2])
        if keep_scores:
            collected.append(torch.stack(values[1:3], dim=1).cpu().numpy())
    averages = (totals / rows).cpu().tolist()
    return averages, np.concatenate(collected) if keep_scores else None


def fit(kind, config, train, validation, device, seed, destination):
    torch.manual_seed(seed)
    model = new_model(kind, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_batches = loader(train, config["batch_size"], seed)
    val_batches = loader(validation, config["batch_size"])
    history = []
    best = float("inf")
    for epoch in range(1, config["epochs"] + 1):
        start = time.monotonic()
        model.train()
        beta = config["beta"] * min(1.0, epoch / config["warmup_epochs"])
        rows = 0
        sums = None
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            values = batch_statistics(model, batch, device, beta)
            loss = values[0].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                total = torch.stack([value.double().sum() for value in values])
                sums = total if sums is None else sums + total
            rows += len(batch[2])
        validation_means, _ = score(model, val_batches, device, config["mc_samples"])
        train_means = (sums / rows).cpu().tolist()
        metric_names = ("loss", "reconstruction", "negative_elbo", "kl") if kind == "vae" else ("bce", "logit")
        entry = {
            "epoch": epoch, "beta": beta if kind == "vae" else None,
            "train": dict(zip(metric_names, train_means)),
            "validation": dict(zip(metric_names, validation_means)),
            "seconds": round(time.monotonic() - start, 2),
        }
        history.append(entry)
        objective = validation_means[0]
        if objective < best:
            best = objective
            torch.save({"state": model.state_dict(), "config": config, "epoch": epoch,
                        "seed": seed, "kind": kind}, destination)
        print(f"{destination.stem} epoch {epoch}/{config['epochs']} "
              f"validation={objective:.5f} time={entry['seconds']:.1f}s", flush=True)
        save_json(destination.with_suffix(".json"), history)
    return best


def features(frame):
    return frame[FEATURES].to_numpy(dtype=float)


def experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True)
    frame, audit = prepare(args.benign_csv, args.malicious_csv, args.split_seed)
    save_json(args.output / "split_audit.json", audit)
    frame[["row_id", "label", "domain", "split"]].to_csv(args.output / "splits.csv.gz", index=False)
    print(json.dumps(audit["splits"], indent=2), flush=True)
    train, validation, test = [frame.loc[frame.split == split].sort_values("length", kind="stable")
                               for split in ("train", "validation", "test")]
    benign_train = train.loc[train.label == 0]
    benign_validation = validation.loc[validation.label == 0]
    floors = Floors(features(benign_train))
    save_json(args.output / "floors.json", {"features": FEATURES, "mean": floors.mean.tolist(),
              "scale": floors.scale.tolist(), "precision": floors.precision.tolist()})
    if args.prepare_only:
        return

    models = args.output / "checkpoints"
    models.mkdir()
    base = {key: getattr(args, key) for key in ("embedding", "hidden", "latent", "beta",
            "epochs", "batch_size", "warmup_epochs", "lr", "mc_samples")}
    plan = {
        "defaults": base, "seeds": args.seeds, "split_seed": args.split_seed, "device": str(device),
        "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
        "python": platform.python_version(), "system": platform.platform(),
        "max_characters": MAX_CHARS,
        "validation_selection": "benign-only beta=1 negative ELBO per character",
        "tune": not args.skip_tuning,
    }
    save_json(args.output / "run_plan.json", plan)
    candidates = [base]
    if not args.skip_tuning:
        candidates.extend([{**base, "latent": value} for value in args.latent_candidates if value != base["latent"]])
        candidates.extend([{**base, "beta": value} for value in args.beta_candidates if value != base["beta"]])
    search = []
    for index, candidate in enumerate(candidates):
        checkpoint = models / f"tune_{index}"
        objective = fit("vae", candidate, benign_train, benign_validation, device,
                        args.seeds[0], checkpoint.with_suffix(".pt"))
        search.append({"config": candidate, "objective": objective, "checkpoint": checkpoint.name})
    selected = min(search, key=lambda candidate: candidate["objective"])
    save_json(args.output / "selection.json", {"candidates": search, "selected": selected})
    trained = {"vae": {args.seeds[0]: models / (selected["checkpoint"] + ".pt")}, "gru": {}}
    for kind in ("vae", "gru"):
        config = selected["config"] if kind == "vae" else base
        fit_train, fit_val = (benign_train, benign_validation) if kind == "vae" else (train, validation)
        for seed in args.seeds:
            if seed in trained[kind]:
                continue
            destination = models / f"{kind}_s{seed}.pt"
            fit(kind, config, fit_train, fit_val, device, seed, destination)
            trained[kind][seed] = destination

    # Test scoring begins only after every checkpoint and headline score is fixed.
    scores = []
    for cohort in (validation, test):
        result = dict(zip(("floor_a", "floor_b"), floors.scores(features(cohort))))
        batches = loader(cohort, args.batch_size)
        for kind, checkpoints in trained.items():
            for seed, path in checkpoints.items():
                model_state = torch.load(path, map_location=device, weights_only=True)
                model = new_model(kind, model_state["config"], device)
                model.load_state_dict(model_state["state"])
                _, model_scores = score(model, batches, device, args.mc_samples, keep_scores=True)
                if kind == "vae":
                    result[f"vae_reconstruction_s{seed}"] = model_scores[:, 0]
                    result[f"vae_negative_elbo_s{seed}"] = model_scores[:, 1]
                else:
                    result[f"supervised_gru_s{seed}"] = model_scores[:, 0]
        if not all(np.isfinite(values).all() for values in result.values()):
            raise RuntimeError("Nonfinite score encountered.")
        scores.append(result)
        cohort[["row_id", "label"]].assign(**result).to_csv(
            args.output / f"{cohort.split.iloc[0]}_scores.csv.gz", index=False)
    report = evaluate(validation, test, *scores, args.seeds, args.bootstrap_repeats, args.bootstrap_seed)
    save_json(args.output / "evaluation.json", report)
    for name, metrics in report["overall"]["metrics"].items():
        print(f"{name:30} AUROC={metrics['auroc']['estimate']:.4f} "
              f"AP={metrics['auprc_ap']['estimate']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("benign-csv", "malicious-csv", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=275)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 97])
    parser.add_argument("--embedding", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--latent-candidates", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--beta-candidates", type=float, nargs="+", default=[0.01, 1.0])
    parser.add_argument("--skip-tuning", action="store_true", help="Use supplied latent size and beta.")
    parser.add_argument("--prepare-only", action="store_true", help="Audit the split and fit the two floors.")
    args = parser.parse_args()
    positive = (args.threads, args.embedding, args.hidden, args.latent, args.epochs, args.warmup_epochs,
                args.batch_size, args.lr, args.mc_samples, *args.latent_candidates)
    if any(value <= 0 or not np.isfinite(value) for value in positive):
        parser.error("Training dimensions and settings must be finite and positive.")
    if any(beta < 0 or not np.isfinite(beta) for beta in (args.beta, *args.beta_candidates)):
        parser.error("Beta values must be finite and nonnegative.")
    if (len(args.seeds) != len(set(args.seeds)) or args.bootstrap_repeats < 20 or
            min(args.split_seed, args.bootstrap_seed, *args.seeds) < 0):
        parser.error("Use distinct nonnegative training seeds and at least 20 bootstrap draws.")
    if args.output.exists():
        parser.error("--output must be a fresh directory to avoid mixing runs.")
    with threadpool_limits(limits=1, user_api="blas"):
        experiment(args)


if __name__ == "__main__":
    main()

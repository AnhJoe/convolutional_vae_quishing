# convolutional_vae_quishing
## URL-only quishing experiment

This branch evaluates a benign-only character GRU VAE against two benign-only
feature floors and a supervised GRU on the URLs encoded by the CIC-Trap4Phish
QR subset. Use the shared experiment runner from the command line or open
[the modeling notebook](notebooks/modeling.ipynb).

### Run the experiment

Use Python 3.11 or later and a CUDA-enabled PyTorch installation for the full
experiment. Install the project dependencies with `python -m pip install -r requirements.txt`.

Get [the team's QR archive](https://drive.google.com/drive/folders/1i5dQyllOAbZAM4XvXrJ0d1MuUuJMGVeb?usp=sharing)
and extract just the two `all_generated_urls_*.csv` files, one from
`QR_All_benign` and one from `QR_All_Malicious`. The runner assigns labels from
the class-specific input arguments and reads the `url` column. It processes
the strings offline and never visits the destinations. Keep the inputs and
outputs in the ignored `data/raw/` and `results/` folders.

```sh
python -m src.run_url \
  --benign-csv data/raw/benign_urls.csv \
  --malicious-csv data/raw/malicious_urls.csv \
  --output results/url_experiment
```

Replace the filenames with your extracted CSV paths. The runner requires a fresh
output directory. Append `--prepare-only` to audit the shared split and fit the
floors before allocating GPU time. Use a separate output directory for that
audit. For a reduced pipeline check, use separately sampled input CSVs and
`--epochs 1 --skip-tuning --seeds 17 --mc-samples 1 --bootstrap-repeats 20`.
Reduced runs cannot establish the five-seed criterion.

### Design

- **Cohort.** Retain the raw strings for tokenization. Remove blanks, every row
  with an exact URL shared by both labels, and redundant exact copies within a
  label. Group both classes together by registrable domain from a pinned,
  offline Public Suffix List snapshot, including private suffixes. IP literals
  use their canonical IP. Unknown suffixes use a last-two-label fallback, and
  unparseable hosts share one group. Assign large domains to training first within the class quotas, then
  balance the remaining indivisible domains toward 70/15/15 per class. The prior class-wise allocator permits cross-class leakage. This
  runner checks every domain across all three splits and records the actual
  counts, prevalence, cleaning, and truncation by split and class.
- **VAE.** Use 95 printable ASCII characters plus an unknown symbol and
  separate padding and decoder-start tokens. Keep up to 256 raw characters.
  A packed GRU encoder predicts a diagonal Gaussian. The teacher-forced GRU
  decoder sees a sampled latent vector at each step. Train with Adam and
  linearly anneal beta from zero to its target over the warmup epochs.
  Select the checkpoint and tune latent size and target beta one factor at a
  time using only benign validation, minimizing beta=1 negative ELBO per
  retained character. Save reconstruction, negative ELBO, and unnormalized KL
  per URL by epoch. Near-zero KL can signal posterior collapse. Average four
  posterior draws by default for held-out reconstruction and negative ELBO.
- **Floors and reference.** Floor A is the absolute benign-train z-score of
  full raw URL length. Floor B is the Mahalanobis distance on length,
  scheme presence, IP-host flag, host-label count, path length, and ASCII digit
  ratio. Floor B uses a small ridge on the standardized benign covariance.
  These floors see the complete strings, including tails beyond the neural
  input cap. The supervised model uses the same GRU encoder architecture with
  one logit and fits from scratch on both training classes. It receives more
  training rows as well as labels. Its performance is a descriptive reference.
- **Held-out scoring.** Prespecify reconstruction NLL from the first seed as
  the headline score. Report negative ELBO separately and summarize both
  scores across five training seeds. Report AUROC, average precision
  (AUPRC), recall at validation-calibrated 1% and 5% FPR, and achieved test
  FPR. Thresholds use the empirical inverse CDF of benign validation scores.
  Flag strictly greater scores, so ties are negative. Fixed thresholds are
  retained during paired test-domain bootstrap resampling. The default
  uses 1,000 draws and percentile 95% intervals, omits class-degenerate
  draws, and withholds intervals with fewer than 20 valid draws.
- **Shortcut checks.** Compute length-decile boundaries from validation,
  merging tied boundaries. Report length and scheme strata with class and
  domain counts. A stratum is eligible with at least 100 rows and 10 domains
  per class in validation, and needs the same support in test to estimate
  metrics. Report IP-host share by class separately. Prespecified success
  requires a positive lower paired CI for headline AUROC vs Floor A overall
  and in a majority of eligible strata, plus a positive worst-seed point
  difference across five seeds. Floor B paired deltas and the fraction of
  the Floor A-to-GRU AUROC gap closed provide additional context. Domain
  separation and these floors control specific shortcuts. Collection and
  source artifacts can still influence results.

### Run outputs

`split_audit.json` and `splits.csv.gz` record input checksums, cleaning,
domain assignments, and class counts. `floors.json` contains the fitted
parameters. `run_plan.json`, `selection.json`, and `checkpoints/` record the
settings, search, training histories, and saved models. The held-out score
files use source-row identifiers. `evaluation.json` contains the test
metrics, per-model thresholds, paired confidence intervals, strata, and
seed summaries.

### Verification

The full QR CSV pair yielded 1,005,700 usable rows after removal of 38
exact cross-label conflicts. The split audit found zero shared domain keys.
Train, validation, and test have 703,990, 150,855, and 150,855 rows,
respectively, each about 57.25% malicious. The largest-domains-first policy
concentrates multi-row malicious domains and all unparseable-host rows in
training. The test set consequently represents mostly singleton malicious
domains with parseable hosts. This is a limitation when generalizing to other
domain mixes or malformed URLs.

A separate 12,000-row, two-epoch pilot on real URLs exercised the validation
search, five training seeds, and paired domain bootstrap. First-seed AUROCs
were 0.669 for VAE reconstruction, 0.529 for Floor A, 0.834 for Floor B, and
0.806 for the supervised GRU. These are reduced-run checks. Full-cohort
five-seed training and held-out metrics require the full experiment command.

## References
Canadian Institute for Cybersecurity. (2025). CIC-Trap4Phish 2025 [Data set]. University of New Brunswick. https://www.unb.ca/cic/datasets/trap4phish2025.html

Nejati, F., Rabbani, M., Eskandarian, M., Mirani, M., Piya, G., Opushnyev, I., Ghorbani, A. A., & Dadkhah, S. (2026). CIC-Trap4Phish: A unified multi-format dataset for phishing and quishing attachment detection. arXiv. https://arxiv.org/abs/2602.09015

Marty, D.-O., Farooqi, S., & Starov, A. (2026, February 13). Phishing on the edge of the web and mobile using QR codes. Unit 42, Palo Alto Networks. https://unit42.paloaltonetworks.com/qr-codes-as-attack-vector/

Puig, A. (2023, December 6). Scammers hide harmful links in QR codes to steal your information. Federal Trade Commission, Consumer Advice. https://consumer.ftc.gov/consumer-alerts/2023/12/scammers-hide-harmful-links-qr-codes-steal-your-information

Almousa, H., & Alsuhibany, S. A. (2026). QR code security: An adaptive retraining approach for dynamic URL-based threat detection. Scientific Reports, 16, Article 13143. https://doi.org/10.1038/s41598-026-43002-z

Bouman, R., & Heskes, T. (2025). Autoencoders for anomaly detection are unreliable. arXiv. https://arxiv.org/abs/2501.13864

Trad, F., & Chehab, A. (2025). Detecting quishing attacks with machine learning techniques through QR code analysis. arXiv. https://arxiv.org/abs/2505.03451

## Team Information:
Joe Nguyen
Adithya Rajendra
Anishwar Chakraborty
Atharva Deshpande
Swaroop Veerla



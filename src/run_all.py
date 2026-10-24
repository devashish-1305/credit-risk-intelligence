"""Run the full pipeline: ingest -> master -> analytics -> model -> validation.

    python -m src.run_all

Needs the 5 raw CSVs in data/raw/. Writes everything to data/processed/.
"""
import sys

import pandas as pd

from src import analytics, ingest, master, model, validation
from src.ingest import PROCESSED_DIR, RAW_DIR

RAW_FILES = ["customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"]


def missing_raw(raw_dir=RAW_DIR):
    return [f"{name}.csv" for name in RAW_FILES if not (raw_dir / f"{name}.csv").exists()]


def main():
    missing = missing_raw()
    if missing:
        sys.exit(f"Missing raw data: {', '.join(missing)}.\n"
                 f"Put all 5 CSVs ({', '.join(f + '.csv' for f in RAW_FILES)}) in {RAW_DIR}/ and run again.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 ingest")
    clean = ingest.clean_all(out_dir=PROCESSED_DIR)
    print("    " + ", ".join(f"{k} {len(v):,}" for k, v in clean.items()))

    print("2/5 master")
    m = master.build_master(master.load_clean())
    m.to_parquet(PROCESSED_DIR / "master.parquet", index=False)
    print(f"    {len(m):,} accounts, default rate {m['target_default'].mean():.1%}")

    print("3/5 analytics")
    top = analytics.build_profiles(m)["employment"].sort_values("relative_risk", ascending=False).iloc[0]
    print(f"    highest-risk employment type: {top.name} ({top['relative_risk']:.2f}x portfolio default rate)")

    print("4/5 model")
    summary = model.save_artifacts(model.run(m))
    for name, met in summary["metrics"].items():
        print(f"    {name:9s} AUC {met['auc']:.3f}  KS {met['ks']:.3f}  Gini {met['gini']:.3f}")

    print("5/5 validation")
    v = validation.save(validation.run(m, pd.read_parquet(PROCESSED_DIR / "test_scores.parquet")))
    for name, b in v["bootstrap"].items():
        print(f"    {name}: AUC diff {b['diff']:+.3f} (95% CI {b['ci_low']:+.3f} to {b['ci_high']:+.3f})")
    print(f"Done. Outputs in {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()

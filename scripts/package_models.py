"""Curate and package PRESTO-R release models and generate SHA-256 manifest."""

import hashlib
from pathlib import Path
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTOR_FILES = [
    "models/predictor/gru_W10_H3.pt",
    "models/predictor/gru_W10_H5.pt",
    "models/predictor/gru_W10_H10.pt",
    "models/predictor/logreg_W10_H5.joblib",
]

AGENT_RUNS = [
    ("predictive_dqn", range(5)),
    ("predictive_dqn_rgg", range(3)),
    ("presto_dqn", range(5)),
    ("presto_dqn_rgg", range(3)),
    ("presto_h10", range(3)),
    ("presto_h3", range(3)),
    ("presto_lam0", range(2)),
    ("presto_lam0.05", range(2)),
    ("presto_lam0.1", range(2)),
    ("presto_lam0.25", range(2)),
    ("presto_lam0.5", range(2)),
    ("presto_oracle", range(2)),
    ("reactive_dqn", range(5)),
    ("reactive_dqn_rgg", range(3)),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    files_to_package: list[tuple[str, Path]] = []

    # 1. Predictors
    for rel_str in PREDICTOR_FILES:
        p = PROJECT_ROOT / rel_str
        if not p.exists():
            raise FileNotFoundError(f"Missing predictor file: {p}")
        files_to_package.append((rel_str.replace("\\", "/"), p))

    # 2. Agent checkpoints + configs
    for method, seeds in AGENT_RUNS:
        for seed in seeds:
            seed_dir = PROJECT_ROOT / "runs" / method / f"seed_{seed}"
            model_p = seed_dir / "best_model.zip"
            cfg_p = seed_dir / "config.yaml"
            if not model_p.exists():
                raise FileNotFoundError(f"Missing model: {model_p}")
            if not cfg_p.exists():
                raise FileNotFoundError(f"Missing config: {cfg_p}")
            rel_model = f"runs/{method}/seed_{seed}/best_model.zip"
            rel_cfg = f"runs/{method}/seed_{seed}/config.yaml"
            files_to_package.append((rel_model, model_p))
            files_to_package.append((rel_cfg, cfg_p))

    print(f"Total files identified for release bundle: {len(files_to_package)}")

    # 3. Compute SHA256 for all files
    checksums: list[tuple[str, str]] = []
    for rel_path, full_path in sorted(files_to_package, key=lambda x: x[0]):
        digest = sha256_file(full_path)
        checksums.append((digest, rel_path))

    # 4. Write models/SHA256SUMS.txt
    manifest_path = PROJECT_ROOT / "models" / "SHA256SUMS.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for digest, rel_path in checksums:
            f.write(f"{digest}  {rel_path}\n")
    print(f"Wrote manifest: {manifest_path} ({len(checksums)} entries)")

    # 5. Create release zip archive
    zip_path = PROJECT_ROOT / "models" / "presto_r_models_v1.0.zip"
    release_info = (
        "PRESTO-R v1.0 Model Release Bundle\n"
        "===================================\n"
        "Contains trained predictor checkpoints and best agent policies evaluated in Phase 5.\n"
        "Regenerating figures and tables from results also requires the separate data/result\n"
        "artifacts listed in README.md.\n"
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("RELEASE_INFO.txt", release_info)
        zf.write(manifest_path, "SHA256SUMS.txt")
        for rel_path, full_path in files_to_package:
            zf.write(full_path, rel_path)

    zip_size_bytes = zip_path.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    zip_sha256 = sha256_file(zip_path)
    print(f"Created archive: {zip_path}")
    print(f"Archive size: {zip_size_bytes} bytes ({zip_size_mb:.2f} MB)")
    print(f"Archive SHA-256: {zip_sha256}")

    # 6. Verify archive extraction round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_p)

        # Check all files
        for digest, rel_path in checksums:
            extracted_file = tmp_p / rel_path
            assert extracted_file.exists(), f"Extracted file missing: {rel_path}"
            extracted_digest = sha256_file(extracted_file)
            assert (
                extracted_digest == digest
            ), f"Checksum mismatch for {rel_path}: expected {digest}, got {extracted_digest}"

    print("All checksums verified successfully upon extraction!")


if __name__ == "__main__":
    main()

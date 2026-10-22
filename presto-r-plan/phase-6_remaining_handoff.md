# Phase 6 Remaining Work: Agent Handoff

## Objective

Finish the outstanding Phase 6 portability and release tasks for PRESTO-R. The
reporting implementation and its generated outputs already exist; focus on verifying
them in a clean environment and preparing a traceable release package.

**Scope restriction:** do not implement, configure, document as completed, or start
Phase 7 / NS-3 work. Phase 7 is optional and explicitly outside this handoff.

## Current state

The following Phase 6 work is already implemented:

- `presto.eval.plots` generates nine report figures as PNG/PDF pairs under
  `results/figures/`.
- `presto.eval.report` generates `results/report.md` from saved metrics and tables.
- README and Makefile contain setup, report, and reproduction instructions/targets.
- Phase 5 metrics and the summary, horizon, robustness, lambda, paired-test, and
  success-criteria outputs have been generated.
- The full test suite and local simulator smoke episode passed in the development
  workspace; Ruff and formatting checks passed for changed code.

Do not redo model training or the completed Phase 5 grids unless a reproducibility
check finds a concrete defect. Do not retune against test-set outputs. Keep the
documented failed success criteria and missing `reactive_dqn_pen` ablation transparent.

## Remaining tasks

### 1. Fresh-environment setup and quick-check verification

**Goal:** demonstrate that the documented setup and quick-check commands work
without relying on this workspace's existing virtual environment or undeclared local
state.

1. Inspect the Git status, tracked files, ignore rules, Python version constraints,
   pinned dependencies, README commands, and Makefile targets.
2. Do not assume the repository is already committed or clean. The current working
   tree may contain untracked project files. Do not stage or commit unrelated files
   or generated results in bulk. If a proper clean clone cannot be created from the
   current Git repository, record that as a blocker and explain what must be committed
   or published first.
3. On a separate clean checkout or second machine, use Python 3.11 and follow the
   README setup instructions exactly:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

4. Verify the quick check:

   ```powershell
   make quick
   ```

   `quick` is currently defined as the test suite followed by the simulator smoke
   episode. Confirm that the smoke image is created at `results/smoke/episode.png`.
   If GNU Make is unavailable, verify the documented equivalent commands
   (`python -m pytest` and `python -m scripts.smoke_env`) and report that Make itself
   was not validated.
5. If setup or quick-check fails, fix only reproducibility issues directly related
   to Phase 6, add/update focused tests when appropriate, and rerun the failing
   command from a clean state. Do not hide missing dependencies or artifacts with
   success-shaped fallbacks.

**Acceptance criteria**

- Record the tested OS, Python version, dependency-install result, exact command
  results, and whether GNU Make was available.
- `make quick` passes on a fresh setup, or its Make-unavailable equivalent passes
  and the limitation is clearly recorded.
- The smoke image is present and non-empty.

### 2. Prepare a curated model release bundle and SHA-256 manifest

**Goal:** make the model artifacts required to interpret/reproduce reported results
available in a clearly named archive or approved release location, with verified
checksums. Avoid copying all experiment data into Git.

1. Inspect the report, configuration files, run configs, and model references to
   identify the minimum supported model bundle. At minimum, assess:
   - Predictor checkpoints referenced by the report/configs, including the
     H=3/H=5/H=10 GRUs and H=5 logistic-regression model where needed.
   - Best agent checkpoints needed for the reported NSFNET/RGG comparisons and
     horizon sweep.
   - Any additional best checkpoints that the supported figure/reproduction commands
     directly load (for example, event-trace policies).
2. Resolve each checkpoint from the corresponding `runs/<method>/seed_<n>/` run
   directory. Include its matching resolved `config.yaml` and any small metadata
   required to identify method, topology/horizon, and training seed. Do not include
   training logs, evaluation arrays, raw parquet grids, predictor `.npz` datasets, or
   unrelated diagnostics unless the release owner explicitly decides they are
   required; these can be very large. The bundle must state that regenerated figures
   also require the separate result/data artifacts listed in the README.
3. Calculate SHA-256 hashes for every file in the curated bundle using a standard
   library or platform tool. Produce a machine-readable manifest (for example,
   `models/SHA256SUMS.txt` or a manifest inside the release archive) containing
   relative file paths and hashes. Recalculate hashes after packaging and verify
   extraction preserves them.
4. Measure and report archive size. Do not claim that the bundle is “a few MB” until
   its actual size has been checked.
5. Attach/upload the archive only to a repository release or shared drive that the
   project owner has authorized. Do not publish private artifacts or credentials.
   If there is no configured remote, release destination, or approval, prepare and
   verify the local bundle and manifest, then clearly report that publishing is
   blocked pending owner action. Do not invent a release URL.
6. Add the chosen bundle location, contents, actual archive size, and verification
   instructions to the README. Add stable direct download links only after a real
   artifact has been published.

**Acceptance criteria**

- Bundle contents are curated and match the model/config references in the report.
- Every included file has a SHA-256 entry; all hashes pass after extraction.
- The archive size and storage location are recorded accurately.
- README tells a user how to obtain, verify, and place/use the artifacts.
- No large raw-result or training-data directory is accidentally committed.

### 3. Export the final report to PDF or Word

**Goal:** provide a submission-ready document derived from the current generated
Markdown report without introducing hand-edited numerical drift.

1. Confirm Phase 5 tables and figures are current before export. Regenerate
   `results/report.md` with:

   ```powershell
   python -m presto.eval.report --metrics results/episode_metrics.parquet --out results/report.md
   ```

2. Export `results/report.md` to PDF or Word using a tool already available in the
   environment (for example, Pandoc or an approved Markdown/document converter).
   Prefer PDF unless the recipient requires Word. Do not add a new project dependency
   just for export without a concrete need.
3. Ensure the export includes all eight referenced inline figure images and all
   report tables. Resolve image paths relative to the Markdown report or bundle
   referenced images alongside the export as required by the chosen exporter.
4. Inspect the rendered document for missing images, clipped tables, broken glyphs
   (including λ and ±), page breaks, and legibility. Fix formatting/configuration,
   not experimental values. Keep `results/report.md` as the generated source.
5. Save the deliverable under `results/`, e.g. `results/report.pdf` or
   `results/report.docx`, and record the exact export command/tool and inspection
   result in this handoff or changelog.

**Acceptance criteria**

- A non-empty PDF or Word file is produced from the latest `results/report.md`.
- All referenced figures/tables are present and readable in the exported document.
- The generated Markdown remains the source of truth for report numbers.

### 4. Create the `v1.0` release tag, only when prerequisites are satisfied

**Goal:** tag the exact committed source/results state associated with the final
report and published model bundle.

1. Inspect `git status`, current branch, remotes, and recent history. The initial
   workspace may not have any commits. Do not tag an uncommitted state, do not
   overwrite an existing tag, and do not use destructive Git operations.
2. If needed, ask the project owner which changes/files belong in the release
   commit. Do not stage all untracked project directories wholesale: this workspace
   contains generated outputs and potentially large artifacts. Exclude virtual
   environments, caches, raw grids, training runs, secrets, and other non-release
   data according to the repository's ignore/release policy.
3. Confirm the intended release commit includes the source, configuration,
   documentation, success criteria, and report artifacts that are meant to be
   versioned. Confirm the report tables and exported document correspond to that
   state. Include the model manifest and a real artifact URL if one was published.
4. Create an annotated `v1.0` tag on the approved commit only after model checksums
   and artifact location are settled. Follow repository commit-message/trailer
   conventions. Do not push the tag unless the owner has authorized publishing.
5. Verify the tag points to the intended commit and that its commit contains the
   expected release files. Record the commit hash and tag verification.

**Acceptance criteria**

- `v1.0` points to the approved commit that produced the reported numbers.
- The tag does not already exist and no commit/tag was created from dirty files.
- The report, model manifest, and any published artifact location are traceable from
  the tagged release.
- If owner approval or a suitable Git remote is unavailable, leave tagging/publishing
  undone and state the precise blocker.

### 5. Update Phase 6 status and release documentation

When work above is finished:

- Update the release checklist and exit criteria in
  `presto-r-plan/phase-6_report.md` to reflect only verified completions.
- Update `CHANGELOG.md` with the actual portability, export, bundle, and tagging
  outcomes; do not describe blocked actions as completed.
- Keep README reproduction and artifact instructions consistent with commands that
  were actually verified.
- Run the focused/full tests and lint/format checks if code or test files changed.
- Run `git diff --check`, verify generated report links and artifacts, and summarize
  completed work, remaining owner actions, test results, and release identifiers.

## Non-goals and guardrails

- Do not implement Phase 7 or any NS-3/ns3-gym integration.
- Do not regenerate completed Phase 5 training/evaluation grids just to recreate
  existing outputs.
- Do not retune from held-out test data or alter/omit failed success criteria.
- Do not commit large raw experiment grids, generated datasets, virtual environments,
  caches, or secrets.
- Do not publish models or results without an authorized destination.
- Do not create a release tag from a dirty working tree or claim a release is
  complete when its upload/tag is blocked.

## Final handoff response

Report:

1. Which checklist items were completed and which remain blocked.
2. Fresh-environment OS/Python, install, quick-check, and Make availability results.
3. Model bundle contents, archive size, manifest verification, and published URL (if
   actually published).
4. Exported document path and visual inspection outcome.
5. Release tag and commit hash (if actually created).
6. Tests/checks run and any outstanding owner action.

# Simplifying the reproduction code

## Recommendation

Keep the paper recipes and make their implementation composable first. The
`stages` proposal identifies a real usability gap, but a generic YAML graph
language would currently add complexity before removing it. A short Python
recipe composed from well-defined operations is a reasonable extension point;
zero Python changes for every future experiment need not be the primary goal.

This review covers the paper's methods and main experiments, the reproduction
and design documents, job builders, schema, adapters, runner, artifact tracking,
selection functions, workflow helpers, CLI, and existing tests. It is not an
audit of the numerical correctness of Bergson or a new GPU reproduction.

## What is already modular

`selection.py` contains small numerical operations. `jobs.py` builds a graph
without executing it. `adapters.py` turns jobs into commands. `executor.py`
runs commands, and `artifacts.py` records results. Those are useful boundaries.
The five experiment builders encode scientific choices worth making visible:
reference seed, source versus target model, query suite, and checkpoint.
Their existence alone is not evidence that a workflow framework is needed.

The low-level CLI already exposes training, attribution, and evaluation
separately. The missing feature is composing those operations in a resumable
manifest without also requesting filtering.

## Problems with the proposed stage format

1. **Ordering edges do not specify inputs.** The adapters read dependency zero
   as a dataset for training, a model for evaluation, or answers for attribution;
   attribution reads dependency one as its model. The proposal's `rank` stage
   depends only on evaluation, so it would not select the reference trained
   model through the existing adapter contract. Generic graph expansion alone
   cannot implement the proposed semantics.
2. **Seed propagation changes the experiment.** The shipped sweeps rank once
   from the first training seed. Depending on a swept baseline evaluation
   without an explicit selector would rank once per seed. Matching an inherited
   `seed` against retraining seeds could also couple reference and retraining
   seeds. Those are different experimental designs.
3. **Multiple inputs need defined joins.** State whether inputs zip, match by
   named dimensions, or cross-product; reject ambiguous matches. Explicit
   source-model and target-model axes are especially important for transfer.
4. **Not every method needs a trained model and judged answers.** Random,
   WildGuard, and rubric scoring do not need the reference evaluation. Loss
   needs a model; gradient attribution needs both model and query. The current
   builders overconstrain these jobs. A new stage API should expose the real
   requirements, not preserve those incidental dependencies.
5. **Analysis has behavior.** It is a command producing `summary.json`, while
   the runner separately collects `manifest.csv` across the shared results
   root. Its scope needs to be explicit, particularly for attribution-only
   runs and shared results from several recipes.

If YAML stages are eventually added, prefer named artifact references such as
`model: baseline.model`, `query: baseline_eval.answers`, and
`dataset: cut.dataset`, plus explicit reference-seed selection. Keep scheduling
edges derived from those references. This is a design direction, not a supported
manifest syntax. Prove it with baseline-only and Figure 1 recipes before adding
arbitrary joins and runtime checkpoint discovery.

## Reproducibility work that matters more than a new syntax

- **Cache validity:** `Job.id` hashes parameters, not input contents, and
  `_job_fingerprint` hashes the job and command arguments. Editing a dataset,
  training template, question file, or script in place does not change those
  arguments. A downstream cache entry also does not include its upstream
  artifact fingerprint. `is_complete` checks metadata, not required outputs.
  Preserve existing artifact names during cleanup, but design versioned input
  fingerprints and upstream invalidation before claiming content-safe resume.
- **Portable planning:** `ResourceConfig` detects GPUs during manifest loading
  and rejects a host with none. Resolve hardware when executing so a laptop
  can validate and inspect a plan. Dry-run purity also deserves attention:
  the standalone `training_commands` helper writes training configs while
  constructing commands.
- **Paper fidelity:** section 3.1 specifies 5,900 training examples and 100
  held out; the reproduction guide describes using all 6,000. Section 3.4
  specifies four initialization seeds crossed with three data shuffles.
  The trainer currently uses one seed for both initialization and an explicit
  dataset shuffle. Twelve values on that axis do not reproduce the factorial
  design. Make the split and both seeds explicit, with provenance, before
  presenting the workflow as an exact reproduction.
- **Dependency provenance:** the guide documents an unpinned Bergson install.
  Record a tested revision and environment versions with the run.
- **Finished outputs:** several figures still lack plotting code. A common
  results loader and small figure functions would help users more immediately
  than a more expressive scheduler.

## What to remove or consolidate

Implemented in this first pass:

- Extract the identical baseline train/evaluate construction used by filter,
  decile, and cross-model sweeps.
- Append analysis once in `build_jobs`, removing five copies.
- Dispatch all five kinds explicitly; remove the implicit training-time fallback.
- Repair existing tests that assumed original-machine archives, an old rubric
  backend, or an obsolete training-directory name. Check checkpoint discovery
  against a temporary filesystem fixture, and plan with explicit mock GPU
  visibility so the manifest tests do not need hardware.

Next candidates, after checking their callers:

- Share train/evaluate expansion across selection recipes.
- Consolidate `PlannedCommand`/`Command` and parallel execution helpers in
  `workflows.py` and `executor.py`; maintain CLI behavior while moving callers.
- Move method-specific attribution command construction out of the large
  adapter branch, with each method declaring its required inputs.
- Share result collection between `iter_stage_artifacts` and
  `write_run_manifest` while preserving the latter's missing-output filtering.

Do not delete `compat.py`, driver scripts, or example pipelines simply because
normal imports do not reference them: subprocess commands and the CLI are
callers too. The archived-checkpoint manifests can be clearly separated as
optional recipes; deleting them would remove functionality, not just dead code.
No production module has been established to be wholly unused in this review.

## Suggested order

1. Land the behavior-preserving cleanup and portable tests.
2. Fix provenance, output validation, and upstream cache invalidation with
   focused mutation/resume tests.
3. Define explicit operation inputs; reuse them in the existing paper recipes
   and a minimal baseline train/evaluate recipe.
4. Add manifest composition only where the resulting user workflow is simpler.
   Avoid maintaining two independent experiment engines indefinitely.
5. Finish reusable figure loaders and plots.

Validation of this first pass: all 24 tests pass. An additional before/after
comparison of all 33 shipped YAML manifests found identical ordered job
records, including IDs, parameters, and dependencies. No training, attribution,
or judging workload was launched. Existing user edits to experiment manifests,
requirements, and the original design proposal were preserved.

## Implementation update

The next increment implements versioned file-content and upstream fingerprints,
required-output content checks, clean reruns with prior artifacts retained under
`.previous`, and blocked-descendant metadata so failed reruns do not publish
stale evaluations. Artifact IDs remain unchanged; old metadata requires a rerun.
Package Python source changes conservatively invalidate cached jobs. Remote model
revisions and installed external environments still need explicit provenance.

Hardware detection now happens at execution time. Training-time planning includes
requested checkpoint observations even when external files are absent. Shared
result collection now uses `iter_stage_artifacts`.

The former catch-all core test module has been separated by responsibility.
Useful numerical and paper-design tests remain, supplemented with CPU subprocess
resume tests and planning tests for every recipe. See `tests/README.md` for the
coverage boundaries. This increment does not introduce a second manifest engine
or yet consolidate the two command/executor APIs.

## Acceptance-first testing revision

The expanded 74-case suite has been reduced to nine offline checks for planning,
selection semantics, and cache correctness. Real acceptance now lives in
`tests/smoke/`: five small YAML workflow templates executed through the production
CLI, with actual model training, generation, judging, and attribution. The suite
checks output coverage and unchanged resume after execution. See `tests/README.md`
for commands and limits; a planning pass alone is not an acceptance pass.

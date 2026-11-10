# Paper experiment plan

## Recommended paper claim

The strongest defensible contribution is **not** generic construction-site
understanding. Excavision contains one identical excavator asset and no workers
or other equipment. The paper should instead claim an excavator-centric,
visibility-aware benchmark and adaptation framework that connects five levels:

1. object detection, counting, and grounding;
2. semantic landmark visibility and occlusion;
3. articulated boom/stick/bucket pose;
4. camera/depth/metric geometry;
5. answerability, uncertainty, and abstention.

The central research question is whether a general vision-language model can be
adapted from image-level description to structured, physically grounded and
calibrated reasoning about articulated construction equipment—and whether
careful curation is more useful than simply adding redundant synthetic frames.

## Proposed contributions

1. **Excavision-QA-5L:** a traceable conversion of exact synthetic scene labels
   into canonical annotations and five levels of construction-equipment visual
   questions, including explicit hidden/out-of-frame states.
2. **Visibility-aware adaptation:** Qwen2.5-VL supervision that distinguishes
   visible, self-occluded, externally occluded/unresolved, and out-of-frame
   landmarks rather than collapsing them into one missing-keypoint flag.
3. **Physically grounded and selective reasoning:** joint evaluation of 2D
   grounding, articulation, metric geometry, and refusal when the supplied
   modalities cannot support an exact answer.
4. **Curation-versus-scale study:** nested 5K/10K/20K/50K/100K data curves and
   matched random controls, with group-safe evaluation, to measure whether
   diversity and difficulty coverage outperform raw volume.
5. **Optional perception-assisted VLM:** compare direct Qwen inference against
   a specialist detector/keypoint front end that supplies structured evidence
   to Qwen. This tests whether a hybrid system is better for deployment than
   forcing one VLM to recover all geometry from pixels.

## Non-negotiable evaluation design

- Split by `split_group_id`/capture before selecting training images. Nearby
  synthetic frames from one scene must never cross train, validation, or test.
- Freeze one validation and one test set for every model and training budget.
- Create OOD test slices by background/HDRI, camera regime, pose, instance
  count, truncation, and occlusion severity.
- Do not claim generalization to workers, other equipment classes, unseen
  excavator designs, or unconstrained real construction sites.
- Use three seeds for principal LoRA comparisons and report confidence
  intervals. One seed is acceptable only for preliminary screening.
- Compare fixed-epoch and fixed-update/compute data-scaling results; otherwise
  larger subsets receive a hidden compute advantage.

## Experiment matrix

### E0 — Data and label audit

Validate canonical/raw agreement, image-label correspondence, all 48 landmark
indices, visibility states, component totals, camera fields, depth encoding,
and derived articulation. Report distributions by source, view, instance count,
occlusion, truncation, depth availability, camera height, and pose. This is a
dataset result, not merely preprocessing.

### E1 — Zero-shot and prompting baselines

Evaluate frozen Qwen2.5-VL-Instruct with:

- zero-shot task prompt;
- few-shot in-context examples;
- constrained JSON output;
- RGB only and RGB+depth where appropriate.

If compute permits, compare at least a small and medium Qwen2.5-VL size. Add a
trivial majority/mean predictor and an annotation-derived oracle to establish
the metric floor and data ceiling.

### E2 — Main LoRA result

Compare the frozen baseline with Qwen2.5-VL LoRA trained jointly on all five
levels. Start with 5K for pipeline validation; run the definitive model on 20K
and 100K. Use identical prompts, output schemas, decoding, validation/test sets,
and evaluation code.

### E3 — Curation versus random selection

Construct random subsets matched to each curated subset for source collection,
depth availability, view, and preferably instance count. The minimum paper set
is 5K, 20K, and 100K with three seeds. The full curve adds 10K and 50K.

Report both fixed-epoch and fixed-update comparisons. This is the experiment
that can support the practical claim that a smaller, diverse subset may be a
better LoRA training set than the complete redundant pool.

### E4 — Five-level capability ablation

Train/evaluate:

- five single-level adapters, L1 through L5;
- one joint balanced five-level adapter;
- a progressive curriculum L1→L2→L3→L4→L5;
- joint training without per-level balancing.

Evaluate every model on every level. The cross-level matrix reveals positive
transfer, interference, and whether geometry/visibility supervision improves
grounding or abstention.

### E5 — Visibility and occlusion ablation

Compare the canonical four-state visibility policy with:

- binary visible/not-visible labels;
- dropping `v=0` points;
- treating every `v=0` as external occlusion;
- removing occlusion labels entirely.

Stratify results by fully visible, self-occluded, external/unresolved,
out-of-frame, and truncated cases. This isolates one of the most distinctive
parts of the dataset.

### E6 — Geometry and modality ablation

Compare:

- RGB only;
- RGB plus aligned depth;
- RGB plus camera metadata;
- RGB, depth, and camera metadata;
- removing 3D-derived supervision;
- depth-shuffled or mismatched controls to detect shortcut learning.

For RGB-only exact-metric questions, the correct target is abstention. Depth
experiments must preserve the encoding metadata; a normalized image without a
metric mapping cannot support an exact metric claim.

### E7 — Direct VLM versus structured perception assistance

Compare:

1. direct Qwen from pixels;
2. specialist detector/keypoint/visibility model alone;
3. specialist predictions serialized as structured evidence for frozen Qwen;
4. specialist evidence plus Qwen LoRA;
5. ground-truth structured evidence plus Qwen as an oracle upper bound.

This experiment separates perception errors from reasoning errors and may be
the strongest systems contribution if direct VLM geometry is weak.

### E8 — Robustness and generalization

Evaluate held-out synthetic groups and controlled degradations: crop/truncation,
resolution, blur, brightness, contrast, depth noise, and increasing occlusion.
Add a small manually annotated real-excavator set if available. Treat it as an
external generalization study, not as proof of broad construction-site
understanding.

### E9 — Calibration and failure analysis

Measure selective prediction before and after calibration. Analyze failures by
view, apparent size, instance count, landmark state, articulation range, depth
availability, and source collection. Include qualitative cases where the model
correctly refuses, hallucinates a hidden landmark, or produces geometrically
inconsistent answers.

## Metrics by level

| Level | Primary metrics | Required stratification |
|---|---|---|
| L1 | count accuracy/MAE, box mAP and AP50/AP75, IoU, valid-JSON rate | size, view, instance count, truncation |
| L2 | four-state landmark macro-F1, per-state recall, component macro-F1, visibility-count MAE | visibility state, component, occlusion severity |
| L3 | angle MAE/median error, within 5°/10°/15°, per-joint error | pose bin, view, visibility of relevant joints |
| L4 | depth MAE/RMSE/relative error when answerable; answerability F1 when not | modality, depth range, camera regime |
| L5 | answerability macro-F1, hallucination rate, ECE/Brier, risk–coverage AUC | hidden/out-of-frame state and difficulty |

Also report schema-valid response rate and a normalized five-level aggregate,
but never replace the task-specific metrics with the aggregate.

## Main paper tables and figures

1. Dataset composition and five-level label distribution.
2. Frozen Qwen versus joint LoRA versus specialist-assisted Qwen.
3. Curated versus random scaling at 5K/20K/100K.
4. Five-by-five curriculum transfer matrix.
5. Visibility-policy and modality ablations.
6. In-distribution versus OOD/real generalization.
7. Calibration and hallucination results.
8. Scaling curves, risk–coverage curves, and qualitative occlusion cases.

## Recommended execution order

1. Freeze group-safe validation/test sets and create matched random controls.
2. Implement deterministic evaluators for all five output schemas.
3. Run frozen Qwen and trivial/oracle baselines.
4. Run a 5K joint-LoRA smoke experiment.
5. Run curated-versus-random 5K screening and select stable hyperparameters.
6. Run definitive 20K and 100K comparisons with three seeds.
7. Run single-level/curriculum, visibility, and modality ablations.
8. Add specialist-assisted Qwen and external real-data evaluation.

The paper should prioritize E2, E3, E4, E5, and E7. A long list of LoRA runs
without fixed group-safe evaluation, random controls, and specialist baselines
would produce a weaker contribution than this focused design.

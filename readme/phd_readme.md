# phd_reference Structure and Gurobi Programs

This document maps the `phd_reference` package structure, the core functions/classes, the notebook purposes, and the Gurobi optimization programs used for worst-case poisoning certification.

## Directory structure

```text
phd_reference/
  setup.py
  data_sets/
    binary_mnist.py
    blobs.py
    cifar.py
    dset_type.py
    halfmoons.py
    hash.py
    hh_anthropic.py
    mnist.py
    perturbed_dset.py
  certifiable_learning_stability/
    agt_certifier.py
    alignment_certifier.py
    bounds.py
    certification_methods.py
    dpa_certifier.py
    gen_stability_certifier.py
    gen_validity_certifier.py
    inference.py
    rdp_certifier.py
    rdp_certify_utils.py
    rs_sampler.py
    solver.py
    threats.py
    models/
      conv.py
      fcn.py
      generic_nn.py
      le_net_5.py
      llm_configs.py
      resnet.py
  experiments/
    metric_transformations.py
    misc.py
    plotting.py
    reproducibility.py
    save_utils.py
    results/
    scripts/
  notebooks/
    agt_playground.ipynb
    alignment.ipynb
    bagging.ipynb
    dpa_playground.ipynb
    dpa_plots.ipynb
    rdp_plots.ipynb
    rdp_stability_playground.ipynb
```

## Package role

`phd_reference` is an experimental package for certifying learning stability under poisoning attacks. It combines:

- standard and perturbed datasets;
- neural network models for tabular/toy/image tasks;
- AGT, RDP, DPA, and LLM alignment certifiers;
- experiment launch scripts and saved YAML results;
- notebook playgrounds for development and plotting;
- Gurobi MILPs for worst-case adversarial allocation across DPA partitions.

The dependency list in `setup.py` includes `gurobipy >= 10.0`, `torch`, `torchvision`, `opacus`, `numpy`, `scipy`, `seaborn`, and `matplotlib`.

## Core modules and functions

### `data_sets/`

- `dset_type.py`
  - `DsetType`: enum for dataset splits such as train/test/validation.

- `mnist.py`
  - `VanillaMNIST`: wraps MNIST.
  - `PerturbedMNIST`: MNIST variant using the shared perturbed dataset interface.

- `binary_mnist.py`
  - `BinaryMNIST`: binary MNIST task.
  - `FeaturePerturbedBinaryMNIST`: feature-perturbed binary MNIST.

- `halfmoons.py`
  - `Halfmoons`: toy half-moons dataset.
  - `PerturbedHalfmoons`: perturbed half-moons dataset.
  - `DiscreteHalfmoons`: discrete perturbation variant.

- `blobs.py`
  - `Blobs`: toy blob classification dataset.

- `cifar.py`
  - `CIFAR`: CIFAR dataset wrapper.
  - `subset(dataset, num_samples)`: returns a smaller CIFAR subset.

- `hh_anthropic.py`
  - `HHAnthropic`: dataset wrapper for Anthropic HH preference/alignment data.
  - `PoisonCombinedTrainHH`: constructs poisoned/combined training preferences.
  - `get_hh_rlhf_preference_dataset(...)`: loads HH preference data.
  - Parsing helpers extract prompts and responses from conversation text.

- `perturbed_dset.py`
  - `PerturbedDataset`: abstract dataset base for label/data perturbation.
  - `FeaturePerturbedDataset`: abstract base for feature perturbations.

- `hash.py`
  - `tensor_generic_hash(data, buckets)`: hashes tensors into bucket indices, used for partitioning.

### `certifiable_learning_stability/`

- `certification_methods.py`
  - `CertificationMethod`: `SGD`, `HYBRID_RDP`, `POINTWISE_RDP`, `AGT`.
  - `RobustnessSetup`: `LOW`, `MEDIUM`, `HIGH`.
  - `AggregationType`: `DPA` and `ROE`.

- `agt_certifier.py`
  - `AGTCertifier`: abstract certifier using Abstract Gradient Training.
  - `certify(...)`: trains/certifies for private budgets `ks_private` and clipping values `clip_gammas`.
  - `vote_and_get_robustness(...)`: loads a bounded AGT partition model, predicts, and returns intrinsic robustness.
  - Dataset-specific certifiers: `MnistCertifier`, `HalfmoonsCertifier`, `BlobsCertifier`, `Cifar10Certifier`.

- `rdp_certifier.py`
  - `StabilityCertifierWithRDP`: abstract certifier using Opacus and Renyi differential privacy.
  - `certify_params(...)`: samples DP-trained parameter intervals and certifies parameter robustness.
  - `certify_points(...)`: certifies pointwise robustness from repeated DP mechanisms and confidence intervals.
  - `hybrid_robustness(...)` and pointwise helpers compute poisoning radii.
  - Dataset-specific certifiers: `HalfmoonsCertifier`, `MnistCertifier`, `Cifar10Certifier`.
  - `validate_and_fix_model(...)`: adapts models for Opacus compatibility.

- `dpa_certifier.py`
  - `StabilityCertifierWithDPA`: abstract certifier for Deep Partition Aggregation.
  - `train_dpa_partitions(...)`: partitions the training set by shard/hash/bagging and trains partition models with selected certification methods.
  - `multi_sample_certification(...)`: casts ensemble votes, computes intrinsic robustness, and calls Gurobi solvers for worst-case accuracy under `k_poison`.
  - `get_metrics_inference(...)`: evaluates DPA/ROE metrics for partition ensembles.
  - `agt_bagging_guarantee(...)`: evaluates AGT-certified bagging guarantees.
  - Dataset-specific certifiers: `HalfmoonsCertifier`, `BlobsCertifier`, `MnistCertifier`, `Cifar10Certifier`.

- `alignment_certifier.py`
  - `AlignmentCertifier`: DPA-style certifier specialized for LLM preference alignment.
  - `train_llm_with_dpo()`: trains each partition with DPO and saves partition models.
  - `generate_tokens_poison(...)` and `generate_responses(...)`: generate partition responses and token/logit outputs.
  - `poison_bench_freq(...)`: measures target entity frequency under clean/poisoned prompts.
  - `_llm_ensemble_vote(...)`: token-level majority voting across partition generations.

- `gen_stability_certifier.py`
  - `LanguageGenerationStabilityCertifier`: extends alignment certification to token/phrase-level stability.
  - `vote_and_get_robustness_column(...)`: computes token-position robustness for generated responses.
  - `multi_sample_robustness_column(...)`: evaluates worst-case token-level stability across poison budgets using DPA or ROE aggregation.
  - Calls `certify_batch_dpa(...)` or `numpy_certify_batch_roe(...)` depending on aggregation mode.

- `gen_validity_certifier.py`
  - `LanguageGenerationValidityCertifier`: validity-oriented extension for language generation experiments.

- `inference.py`
  - `accuracy(...)`: computes model accuracy.
  - `get_prediction(...)`: returns predictions, optionally logits/softmax.
  - `aggregate_predictions_batch(...)`: aggregates predictions/logits for an ensemble over a batch.
  - `aggregate_predictions(...)`: aggregates predictions/logits for full loaders.
  - `get_certified_accuracy_for_given_bounds(...)`: evaluates interval-bound certified accuracy.
  - `aggregate_robustness_radii_to_dict(...)`: converts radii tensors to percentage summaries.

- `rdp_certify_utils.py`
  - Confidence interval helpers: `multi_ci`, `single_ci`, `multi_ci_bagging`, softmax CI variants.
  - Privacy/certification checks: `check_condition_dp`, `check_condition_rdp`, `check_condition_rdp_gp`, bagging variants.
  - Radius searches: `CertifyRadiusDP`, `CertifyRadiusRDP`, `CertifyRadiusBS`, `CertifyRadiusDPBS`, `CertifyRadiusDPBS_softmax_prob`.
  - Privacy accounting: `get_rdp`, `get_cdp`.
  - Experiment helpers: `gen_sub_dataset`, `freeze_params`, `unfreeze_params`.

- `bounds.py`
  - `tightest_interval_for_hit_proportion(...)`: computes tight parameter intervals containing a target hit proportion.

- `threats.py`
  - `Threats`: enum for `L0` and `L2`.
  - `Constraints`: dataclass for f-divergence constraints; imports Gurobi nonlinear functions for default KL divergence, though the active solver models in `solver.py` are MILPs.

- `models/`
  - `Generic_NN`: abstract model base with interval propagation support.
  - `FCN`, `ConvNet`, `LeNet5`, `Resnet18`, `Resnet18Finetune`: task models.
  - `llm_configs.py`: LLM config enums/classes for Gemma, OLMo, Qwen alignment experiments.

### `experiments/`

- `save_utils.py`: result paths, YAML writing/loading, model state saving/loading, torch save/load.
- `reproducibility.py`: seed/device utilities.
- `plotting.py`: plotting functions for CIFAR, alignment, phrase-level stability/validity, AGT bagging, and worst-case accuracy curves.
- `misc.py`: dummy hyperparameter dictionaries for SGD/RDP/AGT playgrounds.
- `metric_transformations.py`: result metric transformation helper.
- `scripts/`: executable experiment definitions for alignment, CIFAR, blobs, halfmoons, and ResNet pretraining.
- `results/`: YAML outputs and parameter files from prior experiments.

## Notebook purposes

- `agt_playground.ipynb`
  - Minimal AGT playground on half-moons.
  - Instantiates `HalfmoonsCertifier` from `agt_certifier.py`.
  - Runs `certify(...)` over several `k_private` values and clipping gammas.

- `rdp_stability_playground.ipynb`
  - RDP certification playground for half-moons, MNIST, and CIFAR-10.
  - Runs `certify_params(...)` for parameter-interval robustness and `certify_points(...)` for pointwise robustness.
  - Uses Opacus/RDP hyperparameters such as `sigma`, `sample_rate`, `max_grad_norm`, and `mechanism_samples`.

- `rdp_plots.ipynb`
  - Analytical/plotting notebook for confidence intervals and RDP radius behavior.
  - Uses beta confidence intervals, Opacus RDP accounting, and `make_multiline_plot(...)`.
  - Explores how sample count, confidence, hit ratio, and radius affect the certification inequalities.

- `dpa_playground.ipynb`
  - Main DPA and ROE playground for CIFAR-10 plus later half-moons AGT bagging.
  - Trains DPA partition ensembles with `train_dpa_partitions(...)`.
  - Runs DPA and ROE inference metrics with `get_metrics_inference(...)`.
  - Compares vanilla SGD partitions, pointwise RDP partitions, bare ResNet, and fine-tuned ResNet setups.

- `bagging.ipynb`
  - Bagging-specific experiments for half-moons and blobs.
  - Trains AGT-certified partition ensembles using `partitioning_method="bag"`.
  - Evaluates `agt_bagging_guarantee(...)` over private budgets and clipping values.

- `alignment.ipynb`
  - LLM alignment certification playground.
  - Instantiates `AlignmentCertifier` over HH Anthropic preference data.
  - Generates partition responses, computes DPA/ROE robustness columns/rows, runs multi-sample robustness, and measures poisoned entity frequency.
  - Uses LLM configs such as `LlmType.GEMMA2B`.

- `dpa_plots.ipynb`
  - Result plotting notebook.
  - Calls plotting utilities for CIFAR, alignment, phrase-level stability/validity, multi-model alignment, worst-case accuracy, and AGT bagging plots.

## Gurobi optimization programs

All active optimization programs are in `certifiable_learning_stability/solver.py`. They are mixed-integer linear programs, not pure linear programs, because they use binary variables and Gurobi indicator constraints. The continuous poison allocation variables are intentionally relaxed to continuous values for faster solving, with comments indicating this gives the same result for these formulations.

### Shared solver setup

- `get_gurobi_license_params()`
  - Reads `certifiable_learning_stability/gurobi.lic`.
  - Passes license parameters into `gp.Env(...)`.

Each MILP creates:

- a Gurobi environment and model named `"Worst Case Adversarial Allocation Vector"`;
- `LogToConsole = 0`;
- `OptimalityTol = 1e-5`;
- a time limit of 300 seconds for Torch-based DPA/ROE solvers and 20 seconds for `numpy_certify_batch_roe(...)`.

The common output is worst-case accuracy:

```text
worst_case_accuracy = 1 - maximum_fraction_of_flipped_predictions
```

If the model hits the time limit, the code returns `1 - model.ObjBound` instead of `1 - model.objVal`.

### Shared variables and constraints

Most solvers use:

- `p[i]`: continuous poisoning allocation for partition/model `i`, lower bounded by 0.
- `z[k]`: binary indicator that datapoint `k` is flipped by the adversarial allocation.
- `z_k[...]`: binary inner indicators that a partition-level vote or case flips once `p[i]` exceeds a robustness threshold.

Common constraints:

```text
sum_i p[i] == k_poison
p[i] <= partition_dset_size    for every partition i
```

Common objective:

```text
maximize (1 / num_datapoints) * sum_k z[k]
```

This chooses the worst adversarial distribution of a fixed poisoning budget across ensemble partitions.

### `certify_batch_targeted_attacks(...)`

Purpose:

- Computes worst-case accuracy for targeted attacks where the adversary tries to avoid certain classes and reduce selected classes.

Inputs:

- `prediction_per_partition`: predicted class per datapoint and partition.
- `aggregation_margins`: per-datapoint margin that must be overcome.
- `avoid_classes`: classes that should not be changed into.
- `reduce_classes`: class set whose votes can be reduced.
- `partition_dset_size`, `k_poison`, `device`.

Program:

- Builds intrinsic robustness `b[i][k]`.
- If partition `i` predicts the avoid class for datapoint `k`, threshold is `k_poison + 1`, making it impossible within budget.
- Otherwise threshold is `1`.
- Indicator `z_k[i] = 1` when `p[i] >= b[i][k]`.
- A prediction is flipped when the weighted count of affected votes exceeds the aggregation margin:

```text
aggregation_margins[k] <= sum_i z_k[i] * (1 + 1{prediction_i is in reduce_classes[k]})
```

### `certify_batch_dpa(...)`

Purpose:

- Computes worst-case batch accuracy for standard DPA majority voting.

Inputs:

- `intrinsic_rob`: per-partition/per-datapoint robustness from AGT/RDP/SGD partition certification.
- `prediction_per_partition`: partition predictions, shaped as datapoints by partitions.
- `batch_labels`: ground-truth labels.
- `num_classes`, `partition_dset_size`, `k_poison`, `device`.

Program:

- Computes class vote counts for each datapoint.
- Finds DPA winner `c_pred` and runner-up `c_sec` by stable vote sorting.
- If `c_pred` is the correct label, the aggregation margin is:

```text
G[k] = votes[c_pred] - votes[c_sec] + 1{c_sec > c_pred}
```

- If prediction is already wrong, `G[k] = -1`, so it is counted as not robust.
- For each partition:
  - if it predicts `c_sec`, threshold is `k_poison + 1`;
  - otherwise threshold is `intrinsic_rob[i][k] + 1`.
- A flipped vote indicator is active when `p[i] >= threshold`.
- A prediction flips when:

```text
G[k] <= sum_i z_k[i] * (1 + 1{prediction_i == c_pred})
```

The extra `1{prediction_i == c_pred}` accounts for both removing a vote from the winner and allowing the runner-up to catch up.

### `dpa_roe_prediction(...)`

Purpose:

- Computes the DPA+ROE winning class and round-two margins before the ROE MILP is built.

Inputs:

- `votes`: vote counts per class.
- `scores`: per-partition class scores/logits.
- `num_classes`, `ensemble_size`.

Logic:

- Takes the top three classes by DPA votes.
- Runs a pairwise score election between the top two.
- Handles missing top-k logits using `NOT_IN_TOPK_THRESHOLD = -30000.0`.
- Uses class-index tie-breaking.
- Returns:
  - `top_3_classes = [c_pred, c_sec, c_third]`;
  - `R2`: pairwise round-two margins for `c_pred` against other classes.

`old_dpa_roe_prediction(...)` is an earlier version of the same idea.

### `certify_batch_dpa_roe(...)`

Purpose:

- Computes worst-case batch accuracy for DPA followed by ROE aggregation.

Inputs:

- Same as DPA, plus `logits_per_partition` for round-two comparisons.

Program:

- For each datapoint, computes top classes and ROE margins with `dpa_roe_prediction(...)`.
- Builds four threshold arrays `bs[0..3, i, k]` and four margins `gs[0..3, k]`.
- Uses three high-level failure cases:

```text
case 1: c_pred can be pushed out before round two
case 2: c_sec can beat c_pred in round two
case 3: c_sec can enter/win through the third-class path, requiring two subconditions
```

Margins:

```text
G1 = gap(c_pred, c_sec) + gap(c_pred, c_third)
G2 = ceil((R2[c_sec] + 1{c_sec > c_pred}) / 2)
G3 = gap(c_sec, c_third)
G4 = min_c ceil((R2[c] + 1{c > c_pred}) / 2), excluding c_pred and c_sec
```

If `c_pred` is not the ground-truth label, all `G` values are set to `-1`.

Variables:

- `z_k[0, i]` through `z_k[3, i]`: binary indicators for four threshold types.
- `l_k[0]`, `l_k[1]`, `l_k[2]`: binary indicators for cases 1, 2, and 3.
- `l_k_3[0]`, `l_k_3[1]`: binary indicators for the two subconditions of case 3.

Indicator constraints activate each case when its inner sum crosses the corresponding margin:

```text
l_k[0] = 1 if G1 <= sum_i z_k[0,i] * (1 + 2*1{prediction_i == c_pred})
l_k[1] = 1 if G2 <= sum_i z_k[1,i]
l_k_3[0] = 1 if G3 <= sum_i z_k[2,i] * (1 + 1{prediction_i == c_sec})
l_k_3[1] = 1 if G4 <= sum_i z_k[3,i]
l_k[2] = 1 if l_k_3[0] + l_k_3[1] == 2
z[k] = 1 if l_k[0] + l_k[1] + l_k[2] >= 1
```

The objective remains to maximize the fraction of flipped datapoints.

### `numpy_certify_batch_roe(...)`

Purpose:

- NumPy version of the DPA+ROE MILP, used by language generation stability where logits are assembled as NumPy arrays.

Differences from `certify_batch_dpa_roe(...)`:

- Inputs are NumPy arrays rather than Torch tensors.
- Uses a 20-second time limit.
- Uses `G4 = ceil((R2[c_third] + 1{c_third > c_pred}) / 2)` rather than the minimum over non-winner/non-runner-up classes.
- Does not add intrinsic robustness into the same threshold updates as the Torch version.

## How the Gurobi MILPs are used

1. Partition models are trained by `StabilityCertifierWithDPA.train_dpa_partitions(...)`.
2. During certification, each partition casts predictions and intrinsic robustness on a test batch.
3. The DPA certifier transposes predictions into `[num_datapoints, ensemble_size]`.
4. For each requested `k_poison`, it calls:
   - `certify_batch_dpa(...)` for `AggregationType.DPA`;
   - `certify_batch_dpa_roe(...)` for `AggregationType.ROE`.
5. Language generation stability uses the same idea per token position, calling:
   - `certify_batch_dpa(...)` for token-level DPA;
   - `numpy_certify_batch_roe(...)` for token-level ROE.
6. The MILP returns worst-case batch accuracy for the given poison budget.

## Important caveats

- The Gurobi models are MILPs with indicator constraints, despite being described informally as linear programs.
- A local `gurobi.lic` file is expected next to the package root used by `solver.py`.
- The solvers suppress console logs; detailed values are sent to `loguru`.
- Several notebooks use hard-coded CUDA device indices and paths, so they may need local adjustment before running.
- The alignment code uses external model caches, DPO training, PEFT, W&B, and large LLM checkpoints; it is much heavier than the toy/image certifiers.

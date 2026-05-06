# Token Voting Setup

This note explains how token voting is represented in the toy certificate code and where the closest equivalent appears in `phd_reference`.

Short answer: the toy example explicitly generates a synthetic prompt/token voting tensor with shape `[K, N, L]`. The `phd_reference` code does not create the same synthetic matrix directly, but its language-generation certifiers build the same logical object from real model generations across DPA partitions.

## Toy Representation

In `toy_certificate/data.py`, the core object is `ToyData`.

The main vote tensors are:

```text
stab_votes[k, i, j]
val_votes[k, i, j]
```

Dimensions:

```text
K = number of shards / partitions / ensemble members
N = number of prompt rows
L = number of generated token columns
T = number of possible token IDs in the toy vocabulary
```

So the toy setup is:

```text
              token column j
             0    1    2    ... L-1
prompt i 0  tok  tok  tok
prompt i 1  tok  tok  tok
prompt i 2  tok  tok  tok
...
```

Each shard `k` provides one whole `[N, L]` grid of token votes. Stacking all shards gives:

```text
votes shape = [K, N, L]
```

For example, one shard layer is:

```text
stab_votes[k] =
[
  [4, 3, 2, 1],
  [1, 0, 0, 0],
  [0, 4, 3, 4],
]
```

That means shard `k` predicts token `4` for prompt row `0`, token position `0`; token `3` for prompt row `0`, token position `1`; and so on.

## Stability Votes

Stability votes model generation under the clean autoregressive prefix.

The generator first samples a base clean token grid:

```text
base_token[i, j]
```

Then each shard mostly follows this base grid, with random disagreement controlled by:

```text
delta_stab
```

So for stability:

```text
stab_votes[k, i, j] = base_token[i, j]
```

unless shard `k` disagrees at that cell, in which case it receives some other token ID.

The code then counts token votes per prompt/token cell:

```text
stab_counts[i, j, t] = number of shards voting for token t at cell (i, j)
```

From this it derives:

```text
clean_pred[i, j] = majority token
runner_up[i, j]  = second-ranked token
```

These are used by the stability MILPs.

## Validity Votes

Validity votes model generation under a harmful target prefix.

The toy code chooses a harmful target matrix:

```text
target[i, j] != clean_pred[i, j]
```

Then it builds `val_votes[k, i, j]` separately from `stab_votes`.

The parameters are:

```text
target_bias = chance a shard naturally emits the harmful target token
delta_val   = chance a shard emits noisy non-base behaviour under the harmful prefix
```

The validity count tensor is:

```text
val_counts[i, j, t] = number of shards voting for token t at cell (i, j)
```

The validity MILPs ask how many shards must be poisoned so that the harmful target tokens win in selected rows/columns.

## Influence Mask

The toy setup can also attach:

```text
influence[k, i, j] in {0, 1}
```

This says whether poisoning shard `k` can affect cell `(i, j)`.

Current modes:

```text
dense        every shard influences every prompt/token cell
row-local    a shard influences selected prompt rows
column-local a shard influences selected token columns
```

## How Counts Are Computed

The helper:

```python
compute_counts(votes, T)
```

takes a `[K, N, L]` tensor and produces:

```text
counts shape = [N, L, T]
```

For each prompt row and token column:

```text
counts[i, j] = bincount(votes[:, i, j], minlength=T)
```

So every cell stores a full vote histogram over token IDs.

## Closest Equivalent In `phd_reference`

The closest equivalent is in:

```text
phd_reference/certifiable_learning_stability/gen_stability_certifier.py
```

The key methods are:

```text
LanguageGenerationStabilityCertifier.vote_and_get_robustness_column(...)
LanguageGenerationStabilityCertifier.multi_sample_robustness_column(...)
LanguageGenerationStabilityCertifier.phrase_level_stability(...)
```

These methods do not sample toy token IDs. Instead, they load each trained DPA partition model, generate real responses for each test prompt, tokenize those responses, and then vote over token positions.

The logical mapping to the toy tensor is:

```text
toy K  = self.num_partitions
toy N  = num_test_samples
toy L  = q generated token positions
toy T  = tokenizer.vocab_size
```

In `vote_and_get_robustness_column(...)`, the reference code builds:

```python
all_tokenized_responses[partition_idx][test_sample_idx]
```

This is a list of generated token IDs for one partition and one prompt.

Then, for a fixed horizon `q`, it cuts every response to the first `q` generated tokens:

```python
predicted_tokens_cut_to_q =
    [all_tokenized_responses[partition_idx][test_sample_idx][:q]
     for partition_idx in range(self.num_partitions)]
```

This is the reference equivalent of collecting all shard votes for one prompt row.

It is then converted to a token-position by partition view:

```python
predicted_tokens_cut_to_q = torch.tensor(predicted_tokens_cut_to_q).permute(1, 0)
```

Shape after the transpose:

```text
[q, num_partitions]
```

For each generated token position `q_idx`, it computes:

```python
word_tokens = predicted_tokens_cut_to_q[q_idx]
word_token_votes = torch.bincount(word_tokens, minlength=self.tokenizer.vocab_size)
```

This is directly analogous to the toy:

```python
np.bincount(votes[:, i, j], minlength=T)
```

The code then stores the top three token classes and vote counts:

```text
cs_pred[q_idx, test_sample_idx]
cs_sec[q_idx, test_sample_idx]
cs_third[q_idx, test_sample_idx]

votes_pred[q_idx, test_sample_idx]
votes_sec[q_idx, test_sample_idx]
votes_third[q_idx, test_sample_idx]
```

These are the reference version of:

```text
clean_pred[i, j]
runner_up[i, j]
winner and runner-up counts
```

The reference code stores these arrays as `[q, num_test_samples]`, while the toy code stores equivalent prompt/token arrays as `[N, L]`.

Mapping:

```text
reference q_idx           -> toy token column j
reference test_sample_idx -> toy prompt row i
```

## Response Generation In `phd_reference`

The actual partition responses are generated in:

```text
phd_reference/certifiable_learning_stability/alignment_certifier.py
```

Relevant methods:

```text
generate_responses(...)
generate_tokens_poison(...)
generate_single_response_iterative(...)
```

`generate_responses(...)`:

1. Loads one saved partition model:

   ```text
   partition_{partition_idx}
   ```

2. Reads prompt strings from the HH-RLHF test set:

   ```python
   prompts = test_set.get_as_column("prompt", (start, end))
   ```

3. Adds an instruction and assistant prefix.

4. Calls Hugging Face `generate(...)`.

5. Strips off the prompt tokens and keeps only generated response tokens:

   ```python
   generated_tokens = tok_response[-delta:]
   tokenized_responses.append(generated_tokens.tolist())
   ```

So `all_tokenized_responses` in the certifier is built from real generated response tokens, not synthetic random tokens.

## Prompt Rows In `phd_reference`

The prompt rows come from:

```text
phd_reference/data_sets/hh_anthropic.py
```

`HHAnthropic` loads:

```text
Anthropic/hh-rlhf
```

and processes each preference example into:

```text
prompt
chosen
rejected
```

The `prompt` field is the final human conversation context. In toy terms:

```text
one HH-RLHF prompt = one prompt row i
```

So the reference setup is not a hand-made prompt/token matrix, but after generation it effectively becomes one:

```text
rows    = HH-RLHF prompts
columns = generated token positions
layers  = DPA partition models
values  = tokenizer token IDs
```

## Phrase-Level Reference

`LanguageGenerationStabilityCertifier.phrase_level_stability(...)` is also close to the toy phrase-DPA baseline.

It groups consecutive generated tokens into phrases:

```python
phrase = tuple(predicted_tokens_cut_to_q[partition_idx][start_idx:end_idx])
```

Then it maps each unique phrase to a class ID and counts votes:

```python
phrase_to_class_tokens[phrase] = class_id
class_counts[class_id] += 1
```

This is similar to the toy phrase baseline in `naive_dpa_readme.md`, where a whole length-`L` generated sequence or phrase is collapsed into one atomic class.

## Validity Reference

The closest validity analogue is:

```text
phd_reference/certifiable_learning_stability/gen_validity_certifier.py
```

Relevant methods:

```text
LanguageGenerationValidityCertifier.vote_and_get_robustness_row(...)
LanguageGenerationValidityCertifier.multi_sample_robustness_row(...)
LanguageGenerationValidityCertifier._get_set_rob_radius_against_targeted_attack(...)
```

This code uses the rejected HH-RLHF response, or an explicit `avoid_sentence`, as the harmful target text.

It tokenizes that target:

```python
tokenized_avoid_sentences =
    [self.tokenizer.encode(avoid_sentence, add_special_tokens=False)
     for avoid_sentence in avoid_sentences]
```

Then it performs iterative generation where the prompt is prefixed with progressively longer pieces of the avoid sentence:

```python
prompt + tokenizer.decode(avoid_tokens[:t])
```

For each step, each partition generates the next token or phrase. The code then counts whether the generated phrase matches the target avoid token/phrase and computes robustness against targeted attack.

This is conceptually close to the toy `val_votes` setup:

```text
toy target[i, j]              -> reference avoid token at position j for prompt i
toy val_votes[k, i, j]        -> reference partition k's generated token/phrase at target-prefix step j
toy val_counts[i, j, target]  -> reference count of partitions matching the avoid token/phrase
```

The reference implementation is more procedural because it actually reprompts the model at each target-prefix step.

## Main Difference From The Toy Matrix

The toy code has an explicit rectangular matrix:

```text
votes[K, N, L]
```

The reference code often stores the same information in lists before converting parts of it to tensors:

```text
all_tokenized_responses[partition_idx][test_sample_idx][token_idx]
predicted_tokens_per_partition[test_sample_idx, partition_idx, q_idx]
cs_pred[q_idx, test_sample_idx]
votes_pred[q_idx, test_sample_idx]
```

So the equivalent matrix exists logically, but it is not always materialized as a single variable named `votes`.

If you wanted to convert the reference stability outputs into the toy shape, the conceptual transformation is:

```python
votes = np.zeros((num_partitions, num_test_samples, q), dtype=int)

for k in range(num_partitions):
    for i in range(num_test_samples):
        votes[k, i, :] = all_tokenized_responses[k][i][:q]
```

That `votes` tensor would match the toy shape:

```text
[K, N, L]
```

The toy implementation is therefore a simplified, explicit version of the token-voting structure that the reference repo builds implicitly from real LLM partition generations.

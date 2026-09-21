# Muon as a selectable ARC-TopK optimizer

## Scope

Add `--optimizer muon` beside existing choices in C4, GLUE, and both CIFAR
training scripts. Keep the DDP communication hook selected separately by
`--compressor`. The synthetic objective script has no Adam training path and
is outside this change.

## Update rule

For eligible matrix weights, use the gradient returned by the DDP hook. Update
momentum as `M = mu * M + G`, then orthogonalize `G + mu * M` (Nesterov), using
the five step Polar Express polynomial from dion. Apply decoupled weight decay
with the unadjusted learning rate, and scale the orthogonalized update by
`sqrt(fan_out / fan_in)` by default. Flatten convolution kernels to
`[out_channels, -1]` before orthogonalization. Batch matrices of the same shape
for the orthogonalization call to reduce launch overhead.

Use AdamW within the same optimizer for embedding, normalization, bias, and
output head parameters. Its default betas are `(0.9, 0.999)`. C4 uses the
dion language-model recipe of scalar LR `0.001` and scalar weight decay `0`;
GLUE and CIFAR inherit their task learning rate and decay policy. The optimizer
must support PyTorch schedulers and `state_dict` save/load. Preserve existing
Adam and SGD branches.

## Parameter selection

Apply Muon only to `nn.Linear` and `nn.Conv2d` weights with at least two
dimensions. Exclude modules named `lm_head`, `classifier`, `score`, or `fc`
and all embeddings. Deduplicate tied parameters. Assign every trainable
parameter exactly once.

## Distributed semantics

The current DDP hook reduces or compresses gradients before `optimizer.step()`.
All ranks update their local momentum from the same synchronized gradient.
For groups containing more than one same-shape matrix, distribute Polar Express
work among the DDP ranks and AllGather the resulting updates in parameter order.
Pad groups to a multiple of world size, as in dion's non-sharded path. Groups
with one matrix run locally. A CLI flag allows redundant local orthogonalization
for performance comparison. Before result collectives, compare the used-matrix
layout across ranks and fail collectively if it differs, preventing a mismatched
AllGather sequence. With a lossy compressor, this is an approximate
Muon optimizer because orthogonalization follows the compressed gradient
average. The implementation targets DDP only; FSDP2 matrix sharding requires
a separate AllToAll path.

## Verification

Unit tests cover the update equation, AdamW fallback, matrix selection,
convolution flattening, grouped orthogonalization, distributed result
AllGather, and checkpoint roundtrip.
Syntax checks cover all modified training scripts. If resources permit,
run a short multi-rank DDP check before long experiments. Do not alter the
repository's existing Python or CUDA dependencies.

# `util/treecut/` — answerable/unanswerable item generator

Where the test items come from. It wraps upstream TreeCut to emit matched math word
problems: an answerable one (`z = 0`) and an unanswerable twin (`z = 1`) whose solution
path has one edge cut. Labels are correct by construction, and the two look the same on
the surface, so a monitor cannot tell them apart without actually working the problem.
That surface-invisible difficulty is what makes the tap-vs-black-box separation
meaningful. A test set whose difficulty shows in the prompt yields no separation.

`generate_items(n, seed)` returns a reproducible list with `floor(n/2)` answerable
items and the remainder unanswerable. Upstream's code is imported from the pinned
`third_party/treecut-math` submodule.

## References

- Ouyang, "TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM
  Hallucination Evaluation" ([arXiv:2502.13442](https://arxiv.org/abs/2502.13442)), code
  [j-bagel/treecut-math](https://github.com/j-bagel/treecut-math) (Apache-2.0), dataset
  [jouyang/treecut-math](https://huggingface.co/datasets/jouyang/treecut-math).

## Run it

Nothing to run directly. Its items become test environments in
[`evals/confidence`](../../evals/confidence) and [`evals/basic_mocked`](../../evals/basic_mocked);
see [RUNBOOK Phase 1](../../RUNBOOK.md#phase-1--real-model-slice-the-cot-confidence-tap-local).

"""Confidence evaluation: does a small model reading the chain of thought catch
hallucinations better than a black-box monitor, and does certification preserve it.

  program.py       GemmaProgram (Pi, thinking on)
  taps.py          GemmaMonitor (tap = reads CoT; comparator = answer only)
  environments.py  TreeCut E_gen / E_adapt (the secret test environment)
  run.py           the real-model slice
"""

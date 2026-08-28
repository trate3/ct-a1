"""Financial evaluation: certify an Apollo linear deception probe (white-box tap)
over the insider-trading scenario.

  probe_tap.py            ApolloProbe + ProbeMonitor (reads residual activations)
  insider_trading_env.py  Apollo's insider-trading scenario as E
  thinking_rollouts.py    training-time: detection mask over a Gemma-4 thought channel
  run.py                  the real-model slice (needs the probe + submodule data)

See probe-track.md (repo root) for training the probe and getting the data.
"""

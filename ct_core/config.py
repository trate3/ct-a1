"""Shared constants for the Certified Taps harness."""

# Bumping this invalidates cross-version commits and certificates.
PROTOCOL_VERSION = "ct/0.2"

# Default ports for the enclave and optional untrusted relay.
ENCLAVE_PORT = 7401
RELAY_PORT = 7402

# Fixed public padded plaintext length len_E.
ENV_PLAINTEXT_BYTES = 262_144

# Security parameter for nonces / commitment randomizer (bytes).
NONCE_BYTES = 16
RANDOMIZER_BYTES = 32

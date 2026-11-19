"""Multi-engine reconstruction foundation: registry, adapter contract, execution manager, evidence store.

Design rule shared by every module here: a missing fact stays missing. An engine that is not verifiably available is
reported UNAVAILABLE with the reason; a confidence nobody measured is ``None``; a fallback is recorded, never hidden.
"""

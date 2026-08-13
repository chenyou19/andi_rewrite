"""Cohesive implementation modules used by :mod:`engine.evaluator`.

The package deliberately has no eager re-exports.  ``VolumeEvaluator`` remains
the public stateful entry point while these modules own leaf algorithms and I/O.
"""

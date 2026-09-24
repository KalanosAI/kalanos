"""The analysis pipeline.

Six stages — discovery, loading, mapping, metrics, scoring, reporting —
each importable and testable on its own.
`inference` sits beside them as a library any stage or adapter can call,
rather than a stage of its own.
"""

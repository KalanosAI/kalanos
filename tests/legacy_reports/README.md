# Historical report fixtures

Reports written by earlier Kalanos versions, kept so `load_report`, `inspect` and `compare` can be tested against real schema-6 files.
Each file had its per-episode `streams` removed, since the loader reads only the report header, the gate, and the episode ids and scores.
Every other value is as the original version wrote it, so dataset-level numbers deliberately do not reconcile with the episodes that remain.

| File | Schema | Written by | Source | Episodes kept |
|---|---|---|---|---|
| `aloha_static_towel-43e0cb1-6.3.0-trimmed.json` | 6.3.0 | `43e0cb1` (0.6.3) | `lerobot/aloha_static_towel@13ad96f` | 1 of 50 |
| `aloha_static_towel-048fef2-6.4.0-trimmed.json` | 6.4.0 | `048fef2` (0.6.5) | `lerobot/aloha_static_towel@13ad96f` | 4 of 50 |
| `lerobot_v3_tiny-6.5.0.json` | 6.5.0 | `0f2a986` (mapping overrides) | `tests/fixtures/lerobot_v3_tiny` | 2 of 2 |

The 6.4 file keeps two episodes the gate lists as failing while their scores say `train_ready: true`; the tests read those as the contradictions a legacy report carries.

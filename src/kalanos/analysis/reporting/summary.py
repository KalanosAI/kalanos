"""Shared review groups; member ids retain the complete underlying evidence."""

from collections import defaultdict


def finding_groups(report):
    episodes = {e.id: e for e in report.episodes}
    groups = defaultdict(list)
    for f in report.findings:
        episode = episodes.get(f.episode_id)
        cohort = (
            (episode.adapter, tuple(episode.tasks or ()))
            if episode
            else ("unknown", ())
        )
        groups[
            (f.metric_id, f.stream or "episode", f.consequence.value, cohort)
        ].append(f)
    return [
        {
            "metric": key[0],
            "role": key[1],
            "consequence": key[2],
            "adapter": key[3][0],
            "tasks": list(key[3][1]),
            "count": len(items),
            "episode_count": len({f.episode_id for f in items}),
            "member_ids": [f.id for f in items if f.id is not None],
        }
        for key, items in sorted(groups.items(), key=str)
    ]

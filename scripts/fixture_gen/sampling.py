"""Walk a spec's time grid and sample every entity on it."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import random

# Internal
from fixture_gen.models import Fields, Sample, TimeSeriesSpec


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


logger = logging.getLogger(__name__)


def build_samples(spec: TimeSeriesSpec) -> list[Sample]:
    """Walk a spec's time grid and sample every entity on it.

    One `random.Random(spec.seed)` stream is shared by the grid (for irregular gaps)
    and by every entity's clock jitter and channel draws, in the order the grid yields
    records and entities appear in `spec.entities`.
    That order is load-bearing: it is what makes a regenerated fixture reproduce the
    same bytes as before — entity-then-channel-then-next-gap is the one draw order that
    matches every fixture in the corpus at once, so it must not be reordered.

    Parameters
    ----------
    spec : TimeSeriesSpec
        The fixture to sample.

    Returns
    -------
    list of Sample
        One `Sample` per (grid index, entity) pair not dropped by a
        `DropBurst` pathology,in grid-then-entity order.
    """

    rng = random.Random(spec.seed)
    samples: list[Sample] = []
    dropped = 0

    for i, t_nominal in spec.grid.times(rng):
        for entity in spec.entities:
            drop = entity.drop_burst()
            if drop is not None and drop.covers(i):
                dropped += 1
                continue

            jitter = entity.clock_jitter()
            t_emit = (
                t_nominal + rng.randint(-jitter.max_offset, jitter.max_offset)
                if jitter
                else t_nominal
            )
            theta = spec.grid.phase_at(t_nominal, entity.phase0)

            fields: Fields = {}
            for group in entity.groups:
                values = [channel.sample(theta, rng) for channel in group.channels]
                if group.name is None:
                    for channel, value in zip(group.channels, values, strict=True):
                        fields[channel.name] = value
                else:
                    fields[group.name] = values

            samples.append(Sample(t=t_emit, entity_id=entity.id, fields=fields))

    logger.debug(
        "Sampled %d records for %s (%d dropped)", len(samples), spec.filename, dropped
    )
    return samples

"""What one policy.yaml says: how a measurement turns into points.

See `docs/ARCHITECTURE.md`'s split between:
- policy (`policy.yaml`); and
- fact (`dictionary.yaml`).
This model is that policy's schema; `kalanos.assets.policy` builds one from a file.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum
from typing import Literal

# External
from pydantic import BaseModel, Field, model_validator


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class ScoreMode(str, Enum):
    """How a metric's raw value is reduced before it meets its band."""

    # fmt: off
    DIRECT   = "direct"    # The value itself is graded.
    ABS_DEV  = "abs_dev"   # |value - target| / |target| is graded.
    # fmt: on


class Band(BaseModel):
    """The good/bad boundary a metric's value interpolates between.

    A value at or past `good` scores 100; at or past `bad` scores 0;
    in between, it scores linearly. Either bound may be absent when
    `docs/METRICS.md` only decided one of them.

    Attributes
    ----------
    good : float or None
        The value at or better than which a score of 100 is reached,
        or `None` if undecided.
    bad : float or None
        The value at or worse than which a score of 0 is reached,
        or `None` if undecided.
    """

    good: float | None = None
    bad: float | None = None


class MetricPolicy(BaseModel):
    """One metric's policy: how its value is reduced, then interpolated.

    Attributes
    ----------
    report_only : bool
        Whether this metric is shown but never graded —
        either because its bands are still *to define*, or by design.
    higher_is_better : bool
        Whether a larger value is better, such as SNR.
        Refused alongside `mode: abs_dev`, since a deviation from a target is
        always lower-is-better.
    mode : ScoreMode
        Whether the metric's own value is graded directly,
        or its deviation from a named `target`.
    target : str or None
        The `limits` key `abs_dev` measures deviation against.
        Required when `mode` is `abs_dev`, forbidden otherwise.
    target_source : str or None
        A `DatasetInfo` attribute name. When set, `pipeline.run` fills
        `limits[target]` from it once the adapter's `describe()` runs,
        unless the policy file already declares that limit itself, which
        always wins. Only meaningful alongside `target`.
    weight : float or None
        This metric's share of its family's weighted rollup.
        `None` where `docs/METRICS.md` marks the weight *to define*.
    thresholds : dict[str, Band]
        Bands keyed by taxonomy type, since the same metric can want different
        bands per signal (joint velocity and joint acceleration, for instance).
        The `"default"` key applies to any type without its own entry.
    limits : dict[str, float]
        Deployment-declared values a metric needs to grade against,
        such as a nominal sampling rate —
        the policy owns these because only the deployment knows them,
        and an absent entry means the metric stays ungraded until one is supplied.
    """

    report_only: bool = False
    higher_is_better: bool = False
    mode: ScoreMode = ScoreMode.DIRECT
    target: str | None = None
    target_source: str | None = None
    weight: float | None = Field(default=None, ge=0)
    thresholds: dict[str, Band] = Field(default_factory=dict)
    limits: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _abs_dev_and_target_come_together(self) -> "MetricPolicy":
        """Refuse a `target` without `abs_dev`, or `abs_dev` without one.

        Returns
        -------
        MetricPolicy
            `self`, unchanged, once `mode` and `target` agree.

        Raises
        ------
        ValueError
            If `mode` is `abs_dev` and `target` is unset,
            or `mode` is `direct` and `target` is set.
        """

        if self.mode == ScoreMode.ABS_DEV and self.target is None:
            raise ValueError("mode: abs_dev requires a target")
        if self.mode == ScoreMode.DIRECT and self.target is not None:
            raise ValueError("target is only meaningful under mode: abs_dev")
        return self

    @model_validator(mode="after")
    def _target_source_requires_a_target(self) -> "MetricPolicy":
        """Refuse a `target_source` with no `target` for it to fill.

        Returns
        -------
        MetricPolicy
            `self`, unchanged, once `target_source` implies `target`.

        Raises
        ------
        ValueError
            If `target_source` is set and `target` is not.
        """

        if self.target_source is not None and self.target is None:
            raise ValueError("target_source is only meaningful alongside a target")
        return self

    @model_validator(mode="after")
    def _higher_is_better_never_pairs_with_abs_dev(self) -> "MetricPolicy":
        """Refuse a direction flag on a metric that already has one.

        A deviation from a target is always lower-is-better by construction,
        so `higher_is_better` alongside `abs_dev` would contradict `mode`
        rather than add information to it.

        Returns
        -------
        MetricPolicy
            `self`, unchanged, once the two do not collide.

        Raises
        ------
        ValueError
            If `mode` is `abs_dev` and `higher_is_better` is set.
        """

        if self.mode == ScoreMode.ABS_DEV and self.higher_is_better:
            raise ValueError("higher_is_better has no meaning under mode: abs_dev")
        return self

    @model_validator(mode="after")
    def _bands_order_their_bounds_by_direction(self) -> "MetricPolicy":
        """Refuse a band whose bounds contradict its own scoring direction.

        A band carrying only one bound has nothing to order and is left alone —
        that is exactly `snr_db`'s shape before its `bad` bound is decided.

        Returns
        -------
        MetricPolicy
            `self`, unchanged, once every fully-specified band orders its
            bounds by `higher_is_better`.

        Raises
        ------
        ValueError
            If a band carries both bounds and they are equal,
            or ordered the wrong way for `higher_is_better`.
        """

        for taxonomy_type, band in self.thresholds.items():
            if band.good is None or band.bad is None:
                continue
            if band.good == band.bad:
                raise ValueError(
                    f"{taxonomy_type!r} band's good and bad bounds are equal; "
                    "interpolation needs a real span between them"
                )
            wrong_direction = (
                band.good < band.bad if self.higher_is_better else band.good > band.bad
            )
            if wrong_direction:
                raise ValueError(
                    f"{taxonomy_type!r} band orders good={band.good} and "
                    f"bad={band.bad} the wrong way for "
                    f"higher_is_better={self.higher_is_better}"
                )
        return self


class Policy(BaseModel):
    """One parsed policy.yaml: the grading policy for a deployment.

    Attributes
    ----------
    schema_version : int
        The policy format version,
        so a future breaking change to this shape has something to check against.
    metrics : dict[str, MetricPolicy]
        Policy for each metric, keyed `"family.metric"` — the family a metric
        belongs to and its own name from the registry, joined by one dot.
    family_weights : dict[str, float]
        Each family's share of the overall rollup. Relative, not shares:
        they need not sum to one, since `missing_family` renormalises over
        whichever families a recording actually produced a graded metric for.
    missing_family : {"skip"}
        What to do about a declared family that produced nothing to grade.
        `"skip"` is the only decided behaviour, so it is the only member —
        a second one would be a decision nobody made.
    fail_penalty : float or None
        Points subtracted per distinct metric that scored 0 in a channel,
        or `None` to apply no penalty at all.
    fail_penalty_cap : float or None
        The most `fail_penalty` may subtract from one channel,
        or `None` for no cap.
    letters : dict[str, float]
        The minimum score each of `A`, `B`, `C`, `D` needs;
        below `D`'s minimum is `F`, which carries no minimum of its own.
    """

    schema_version: int
    metrics: dict[str, MetricPolicy]
    family_weights: dict[str, float] = Field(default_factory=dict)
    missing_family: Literal["skip"] = "skip"
    fail_penalty: float | None = Field(default=None, ge=0)
    fail_penalty_cap: float | None = Field(default=None, ge=0)
    letters: dict[str, float]

    @model_validator(mode="after")
    def _every_metric_key_is_one_family_dot_one_name(self) -> "Policy":
        """Refuse a metrics key that is not exactly `family.metric`.

        Checks both the dot count and that neither half is empty —
        `.drop_rate` passes a bare dot count but would silently land the
        metric in family `""`, which no `family_weights` entry can ever match.

        Returns
        -------
        Policy
            `self`, unchanged, once every key splits into two non-empty parts.

        Raises
        ------
        ValueError
            If a key carries zero or more than one dot, or either half is empty.
        """

        malformed = [
            key
            for key in self.metrics
            if key.count(".") != 1 or not all(key.split("."))
        ]
        if malformed:
            raise ValueError(
                f"metrics keys must be exactly 'family.metric', got: {malformed}"
            )
        return self

    @model_validator(mode="after")
    def _bare_metric_names_are_unique_across_families(self) -> "Policy":
        """Refuse two families claiming the same bare metric name.

        The registry keys a metric's result by its bare function name alone,
        with no family attached — `resolve_status` could not tell which
        policy entry a duplicate name meant.

        Returns
        -------
        Policy
            `self`, unchanged, once every bare name is unique.

        Raises
        ------
        ValueError
            If the same bare name appears under more than one family.
        """

        seen_bare_names: set[str] = set()
        collisions = []
        for key in self.metrics:
            bare = key.partition(".")[2]
            if bare in seen_bare_names:
                collisions.append(key)
            seen_bare_names.add(bare)
        if collisions:
            raise ValueError(
                f"metric names must be unique across families, duplicated: {collisions}"
            )
        return self

    @model_validator(mode="after")
    def _letters_are_the_four_grades_in_strict_descent(self) -> "Policy":
        """Refuse a letters table that is not exactly A > B > C > D.

        `F` is the floor below `D`'s minimum and carries no entry of its own.

        Returns
        -------
        Policy
            `self`, unchanged, once the table matches.

        Raises
        ------
        ValueError
            If the keys are not exactly `A`, `B`, `C`, `D`,
            or their minimums do not strictly descend in that order.
        """

        expected_keys = {"A", "B", "C", "D"}
        if set(self.letters) != expected_keys:
            raise ValueError(
                f"letters must carry exactly {sorted(expected_keys)}, "
                f"got {sorted(self.letters)}"
            )
        ordered = [self.letters[grade] for grade in ("A", "B", "C", "D")]
        if ordered != sorted(ordered, reverse=True) or len(set(ordered)) != len(
            ordered
        ):
            raise ValueError(
                f"letters must strictly descend A > B > C > D, got {self.letters}"
            )
        return self

    def entry_for(self, metric_name: str) -> tuple[str, MetricPolicy] | None:
        """Look up one metric's family and policy by its bare registry name.

        Parameters
        ----------
        metric_name : str
            The metric's own name, as the registry knows it —
            the half of a `metrics` key after the dot.

        Returns
        -------
        tuple[str, MetricPolicy] or None
            The metric's family and policy, or `None` if no family declares it.
        """

        for key, metric_policy in self.metrics.items():
            family, _, bare = key.partition(".")
            if bare == metric_name:
                return family, metric_policy
        return None

    def with_limit(self, name: str, value: float) -> "Policy":
        """Return a copy with `name` filled into every abs_dev metric that targets it.

        Skips a metric that already declares `name` under its own `limits`:
        a value the policy file carries always wins over a format's default.

        Parameters
        ----------
        name : str
            The `limits` key to fill in.
        value : float
            The value to fill it with.

        Returns
        -------
        Policy
            A copy with `name` added to every matching metric's `limits`,
            or `self` unchanged when nothing matched.
        """

        updated = {}
        changed = False
        for key, metric_policy in self.metrics.items():
            if (
                metric_policy.mode is ScoreMode.ABS_DEV
                and metric_policy.target == name
                and name not in metric_policy.limits
            ):
                updated[key] = metric_policy.model_copy(
                    update={"limits": {**metric_policy.limits, name: value}}
                )
                changed = True
            else:
                updated[key] = metric_policy

        if not changed:
            return self
        return self.model_copy(update={"metrics": updated})

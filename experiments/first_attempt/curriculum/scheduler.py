"""The scheduler: change the prior from easy to hard as training goes on.

You give it a schedule keyed by global step:

    schedule = {
        0:   {"noise_std": 0.05, "max_features": 20},
        200: {"noise_std": 0.15},
        400: {"noise_std": 0.30},
    }

and call .step(global_step) once per training step (or just iterate it - see
below). When the step crosses a threshold it pushes the new knobs into the prior.

Two kinds of knob, handled by apply_knobs:

  * EXTERNAL - plain attributes on TabICL's generator (feature/class/row counts).
    TabICLPriorDataLoader keeps the generator at loader.pd (a PriorDataset) ->
    loader.pd.prior (an SCMPrior), and reads these off the SCMPrior every time
    it makes a batch, so we set them there.

  * INTERNAL - the sampled hyper-parameters (noise, MLP depth/width, causes).
    These aren't single values, they're *ranges* the prior samples from. We move
    the top of the range (max_mean): low = easy, high = hard. This is where the
    real difficulty lives (features/classes barely move it - see the sweep).
"""

# plain attributes on the generator
EXTERNAL_KNOBS = {"min_features", "max_features", "max_classes", "min_seq_len", "max_seq_len"}
# sampled-HP ranges; the value we set is the upper bound (max_mean) of the range
INTERNAL_KNOBS = {"noise_std", "num_layers", "hidden_dim", "num_causes"}
KNOBS = EXTERNAL_KNOBS | INTERNAL_KNOBS


def apply_knobs(prior, params):
    """Push a {knob: value} dict into a TabICL prior loader, in place."""
    pd = getattr(prior, "pd", None)
    scm = getattr(pd, "prior", None)

    for name, value in params.items():
        if name in EXTERNAL_KNOBS:
            # set it wherever the attribute actually lives
            for target in (prior, pd, scm):
                if target is not None and hasattr(target, name):
                    setattr(target, name, value)
        elif name in INTERNAL_KNOBS:
            if scm is None or not hasattr(scm, "sampled_hp"):
                raise RuntimeError(
                    "this prior can't control internal knobs - build it with "
                    "curriculum.prior.make_prior so it gets a private sampled_hp"
                )
            spec = scm.sampled_hp[name]
            spec["max_mean"] = value
            # keep the range valid: the bottom can't sit above the new top
            if spec.get("min_mean", 0) > value:
                spec["min_mean"] = value
        else:
            raise ValueError(f"unknown knob {name!r}. allowed: {sorted(KNOBS)}")


class CurriculumScheduler:
    def __init__(self, prior, schedule):
        bad = {k for stage in schedule.values() for k in stage} - KNOBS
        if bad:
            raise ValueError(f"unknown curriculum knob(s): {sorted(bad)}. allowed: {sorted(KNOBS)}")

        self.prior = prior
        # keep thresholds sorted so "latest stage at or below the step" is easy to find
        self.schedule = dict(sorted(schedule.items()))
        self.current_stage = None
        # counts how many batches we've yielded over the whole run (across epochs).
        # this is exactly the global step the schedule is keyed on.
        self.global_step = 0

    def step(self, global_step):
        # find the most advanced stage whose threshold we've passed
        stage = None
        for threshold, params in self.schedule.items():
            if global_step >= threshold:
                stage = params

        # only do something when we move into a new stage
        if stage is not None and stage is not self.current_stage:
            self.current_stage = stage
            apply_knobs(self.prior, stage)
            print(f"[curriculum] step {global_step}: -> {stage}", flush=True)

    # The scheduler stands in for the prior in the training loop. As the loop
    # pulls batches, we advance the curriculum ourselves - so the *unmodified*
    # train() loop gets easy->hard data without knowing anything about us.
    # We set the stage first, then generate, so each batch uses the new knobs.
    def __iter__(self):
        prior_iter = iter(self.prior)
        for _ in range(len(self.prior)):
            self.step(self.global_step)
            self.global_step += 1
            yield next(prior_iter)

    def __len__(self):
        return len(self.prior)

    @property
    def num_steps(self):
        return self.prior.num_steps

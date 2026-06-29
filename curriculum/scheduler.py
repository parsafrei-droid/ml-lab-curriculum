"""The scheduler: change the prior from easy to hard as training goes on.

You give it a schedule keyed by global step:

    schedule = {
        0:   {"max_features": 5,   "max_classes": 2},
        200: {"max_features": 20,  "max_classes": 5},
        500: {"max_features": 100, "max_classes": 10},
    }

and call .step(global_step) once per training step. When the step crosses a
threshold it pushes the new knobs into the prior.

The fiddly bit: TabICLPriorDataLoader builds the real generator at init time and
keeps it at loader.pd (a PriorDataset) -> loader.pd.prior (an SCMPrior). The
generator reads these values off the SCMPrior *every time it makes a batch*, so
to actually change difficulty we have to set them down there, not just on the
loader. _apply walks those layers and sets the value wherever it lives.
"""

# The knobs we know how to change mid-training. These are the real attribute
# names on TabICL's generator, so a config can't quietly set something that
# does nothing.
KNOBS = {"min_features", "max_features", "max_classes", "min_seq_len", "max_seq_len"}


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

    def _targets(self):
        """The objects that might hold a knob: the loader, its PriorDataset, and the SCMPrior."""
        pd = getattr(self.prior, "pd", None)
        return [self.prior, pd, getattr(pd, "prior", None)]

    def _apply(self, params):
        for name, value in params.items():
            for target in self._targets():
                if target is not None and hasattr(target, name):
                    setattr(target, name, value)

    def step(self, global_step):
        # find the most advanced stage whose threshold we've passed
        stage = None
        for threshold, params in self.schedule.items():
            if global_step >= threshold:
                stage = params

        # only do something when we move into a new stage
        if stage is not None and stage is not self.current_stage:
            self.current_stage = stage
            self._apply(stage)
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

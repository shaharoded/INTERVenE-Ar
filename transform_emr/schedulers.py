"""
schedulers.py
=============

Unified auxiliary loss weighting scheduler for Phase-1 (embedding) and Phase-2 (transformer) training.

LambdaScheduleController:
  - Accepts a phase-specific schedule config dict (from TRAINING_SETTINGS["phase1_scheduler"] or ["phase2_scheduler"]).
  - Defines auxiliary tasks and their curriculum via an `order` list-of-lists:
      - Each inner list is a stage of aux tasks that activate together.
      - Stage 0 activates immediately at start_epoch.
      - Subsequent stages are unlocked dynamically when the active loss plateaus.
  - Frozen-fraction calibration: lambda_max = fraction_cap * main_loss / aux_loss (computed once, then fixed).
  - Linear ramp from 0 to lambda_max over ramp_epochs (ramp_epochs=1 means immediate, no ramp).
  - Warmup tracking: reports the epoch after which early stopping may begin.

Config expected shape (phase-specific dict):
    {
        "aux_fraction_caps":  {name: fraction, ...},  # required; every aux name must be present
        "order":              [[name, ...], [name, ...], ...],
        "ramp_epochs":        {name: int, ...},
        # Multi-stage only (len(order) > 1):
        "plateau_min_delta":  float,
        "plateau_patience":   int | [int, ...],   # one per stage transition
    }
"""


def linear_schedule(epoch: int, start_epoch: int, end_epoch: int, max_val: float) -> float:
    """
    Linearly ramp from 0 to `max_val` over [start_epoch, end_epoch].
    If end_epoch <= start_epoch, immediately returns max_val at start_epoch (no ramp).
    """
    if epoch < start_epoch:
        return 0.0
    if end_epoch <= start_epoch:
        return max_val
    progress = min(max((epoch - start_epoch) / float(end_epoch - start_epoch), 0.0), 1.0)
    return max_val * progress


class LambdaScheduleController:
    """
    Unified scheduler for auxiliary loss weighting.

    Handles both Phase-1 (embedding) and Phase-2 (transformer) training from a single
    phase-specific config dict. Phase behaviour is driven purely by the `order` list:
      - Single stage  → Phase-1 style: all aux tasks activate immediately, no plateau gating.
      - Multi-stage   → Phase-2 style: stage 0 activates immediately; later stages unlock
                        on plateau detection of all currently active objectives.

    Usage:
      - Once per epoch (after validation): call update(epoch, vl_main, **aux_losses)
          where aux_losses keys are plain aux names (e.g., mlm=0.4, dt=0.1).
      - Per batch: call get_lambdas(epoch) to retrieve current lambda values.
      - Optional logging: call status_line(epoch).
    """

    def __init__(self, schedule_config: dict, start_epoch: int = 0):
        """
        Parameters
        ----------
        schedule_config : dict
            Phase-specific scheduler config (see module docstring for expected keys).
        start_epoch : int
            Current training epoch (for checkpoint resume).
        """
        self._cfg = schedule_config
        self.start_epoch = int(start_epoch)
        self._min_aux_loss = 1e-8
        self._max_lambda_clamp = 100.0

        caps = schedule_config["aux_fraction_caps"]
        ramp_cfg = schedule_config.get("ramp_epochs", {})
        order = schedule_config.get("order", [])
        self._order = order  # [[aux_name, ...], ...]

        # Register all auxiliaries across all stages.
        # Raises KeyError immediately if any aux name is missing from aux_fraction_caps.
        self._auxiliaries = {}
        for stage_idx, stage_auxi in enumerate(order):
            s_epoch = self.start_epoch if stage_idx == 0 else None
            for name in stage_auxi:
                if name not in caps:
                    raise KeyError(
                        f"aux_fraction_caps is missing an entry for '{name}'. "
                        f"Add it explicitly — no silent defaults."
                    )
                self._register_aux(
                    name=name,
                    start_epoch=s_epoch,
                    ramp_epochs=max(1, int(ramp_cfg.get(name, 1))),
                    fraction=caps[name],
                )

        # Multi-stage: set up plateau-based curriculum
        n_stages = len(order)
        self._has_dynamic = n_stages > 1

        if self._has_dynamic:
            patience_cfg = schedule_config.get("plateau_patience", 3)
            if isinstance(patience_cfg, int):
                patience_cfg = [patience_cfg] * (n_stages - 1)

            self.plateau_min_delta = float(schedule_config.get("plateau_min_delta", 1e-4))
            self._plateau_patience = patience_cfg   # one entry per stage transition

            self._current_stage = 0
            self._stage_start_epoch = self.start_epoch
            self._stage_best = float("inf")
            self._stage_bad_epochs = 0

        self._warmup_complete_epoch = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_aux(self, name: str, start_epoch, ramp_epochs: int, fraction: float):
        self._auxiliaries[name] = {
            "name": name,
            "start_epoch": start_epoch,
            "ramp_epochs": max(1, int(ramp_epochs)),
            "fraction": float(fraction),
            "lambda_max": None,
            "anchor_main_loss": None,
            "anchor_aux_loss": None,
        }

    @staticmethod
    def _check_plateau(metric_val, best_val, bad_epochs, min_delta, patience):
        """Returns (new_best, new_bad_epochs, is_plateau)."""
        if metric_val < (best_val - min_delta):
            return metric_val, 0, False
        bad_epochs += 1
        return best_val, bad_epochs, bad_epochs >= patience

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_dynamic(self) -> bool:
        """True when the scheduler has more than one stage (plateau-gated curriculum)."""
        return self._has_dynamic

    def update(self, epoch: int, vl_main: float, **aux_losses) -> list:
        """
        Calibrate auxiliaries and advance dynamic stage transitions.

        Parameters
        ----------
        epoch : int
            Current epoch.
        vl_main : float
            Validation main loss (e.g. BCE).
        **aux_losses : float
            Named keyword args using plain aux names as keys
            (e.g. mlm=0.4, dt=0.1, ce=0.3, penalty=0.2, outcome=0.5).

        Returns
        -------
        list[str]
            Log messages for calibration events and stage transitions.
        """
        messages = []

        # Step 1: Check for stage transitions FIRST so newly unlocked aux can
        # be calibrated in the same call.
        if self._has_dynamic:
            messages.extend(self._check_stage_transitions(epoch, vl_main, aux_losses))

        # Step 2: Calibrate each auxiliary once (when active and loss is available).
        for name, spec in self._auxiliaries.items():
            if spec["start_epoch"] is None:
                continue  # Not yet unlocked
            if spec["lambda_max"] is not None:
                continue  # Already calibrated
            if name not in aux_losses:
                continue  # Loss not provided this call
            vl_aux = float(aux_losses[name])
            if vl_aux > self._min_aux_loss:
                lam = (spec["fraction"] * vl_main) / max(vl_aux, self._min_aux_loss)
                spec["lambda_max"] = min(lam, self._max_lambda_clamp)
                spec["anchor_main_loss"] = vl_main
                spec["anchor_aux_loss"] = vl_aux
                messages.append(
                    f"[Scheduler]: {name} calibrated at epoch {epoch}, "
                    f"λ_max={spec['lambda_max']:.4f} "
                    f"(main={vl_main:.4f}, aux={vl_aux:.4f})"
                )

        return messages

    def _check_stage_transitions(self, epoch: int, vl_main: float, aux_losses: dict) -> list:
        """Check whether the next stage should be unlocked based on plateau detection."""
        messages = []

        # Nothing to do if already at the last stage
        if self._current_stage >= len(self._order) - 1:
            return messages

        transition_idx = self._current_stage
        next_stage_idx = self._current_stage + 1
        next_stage_auxi = self._order[next_stage_idx]

        # If next stage was somehow already unlocked (resume edge case), sync and exit
        if all(self._auxiliaries[n]["start_epoch"] is not None for n in next_stage_auxi):
            self._current_stage = next_stage_idx
            return messages

        # Plateau metric: main loss + all currently active aux losses
        active_auxi = [n for s in self._order[:next_stage_idx] for n in s]
        metric = vl_main + sum(float(aux_losses.get(n, 0.0)) for n in active_auxi)

        self._stage_best, self._stage_bad_epochs, plateau = self._check_plateau(
            metric, self._stage_best, self._stage_bad_epochs,
            self.plateau_min_delta, self._plateau_patience[transition_idx],
        )

        if plateau:
            unlock_epoch = epoch + 1
            for name in next_stage_auxi:
                self._auxiliaries[name]["start_epoch"] = unlock_epoch

            messages.append(
                f"[Scheduler][Dynamic]: Stage {next_stage_idx} "
                f"({', '.join(next_stage_auxi)}) unlocked at epoch {unlock_epoch}"
            )

            # Warmup completes after the last stage's ramp finishes
            if next_stage_idx == len(self._order) - 1:
                max_ramp = max(self._auxiliaries[n]["ramp_epochs"] for n in next_stage_auxi)
                self._warmup_complete_epoch = unlock_epoch + max_ramp
                messages.append(
                    f"[Scheduler]: Warmup completes at epoch {self._warmup_complete_epoch}"
                )

            # Advance tracking state
            self._current_stage = next_stage_idx
            self._stage_start_epoch = unlock_epoch
            self._stage_best = float("inf")
            self._stage_bad_epochs = 0

        return messages

    def get_lambdas(self, epoch: int) -> dict:
        """
        Return current lambda values for all registered auxiliaries.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        dict
            {aux_name: lambda_value}. Returns 0.0 for aux tasks not yet active
            or not yet calibrated.
        """
        lambdas = {}
        for name, spec in self._auxiliaries.items():
            if spec["start_epoch"] is None or spec["lambda_max"] is None:
                lambdas[name] = 0.0
            else:
                start = spec["start_epoch"]
                ramp = spec["ramp_epochs"]
                end = start if ramp <= 1 else start + ramp
                lambdas[name] = linear_schedule(epoch, start, end, spec["lambda_max"])
        return lambdas

    def current_warmup_end_epoch(self):
        """
        Return the epoch after which early-stopping may begin counting.

        Returns
        -------
        int | float | None
            - Multi-stage: float('inf') until last stage unlocked, then concrete epoch.
            - Single-stage: None (caller manages warmup separately).
        """
        if not self._has_dynamic:
            return None
        if self._warmup_complete_epoch is None:
            return float("inf")
        return self._warmup_complete_epoch

    def status_line(self, epoch: int) -> str:
        """Human-readable status line for logging."""
        parts = []
        lambdas = self.get_lambdas(epoch)
        for name in sorted(self._auxiliaries.keys()):
            spec = self._auxiliaries[name]
            lam = lambdas[name]
            if spec["lambda_max"] is None:
                parts.append(f"{name}:λ={lam:.4f}(pending)")
            else:
                parts.append(f"{name}:λ={lam:.4f}/λ_max={spec['lambda_max']:.4f}")
        return f"[Scheduler] epoch={epoch} | {' | '.join(parts)}"

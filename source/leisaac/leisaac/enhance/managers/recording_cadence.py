class RecordingCadence:
    """Track which action steps should be written to a dataset."""

    def __init__(self, action_hz: int, recording_hz: int) -> None:
        if action_hz <= 0 or recording_hz <= 0:
            raise ValueError("action_hz and recording_hz must be positive")
        if action_hz % recording_hz != 0:
            raise ValueError(
                f"action_hz ({action_hz}) must be divisible by recording_hz ({recording_hz})"
            )
        self.interval = action_hz // recording_hz
        self._step_index = 0

    def begin_step(self) -> bool:
        """Advance the action-step counter and return whether to record it."""
        should_record = self._step_index % self.interval == 0
        self._step_index += 1
        return should_record

    def reset(self) -> None:
        self._step_index = 0

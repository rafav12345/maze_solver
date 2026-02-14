"""Non-blocking animation system using Tk root.after() scheduling.

Replaces the original sleep(0.05) blocking pattern. Algorithms yield events,
and the Animator schedules one event per frame using root.after().
"""

from __future__ import annotations

from typing import Callable, Iterator

from tkinter import Tk

from core.events import AlgorithmEvent


class Animator:
    """Schedules algorithm events as non-blocking Tk callbacks.

    Args:
        root: The Tk root window.
        delay_ms: Milliseconds between animation frames.
    """

    def __init__(self, root: Tk, delay_ms: int = 50) -> None:
        self._root = root
        self.delay_ms = delay_ms
        self._event_iter: Iterator[AlgorithmEvent] | None = None
        self._on_event: Callable[[AlgorithmEvent], None] | None = None
        self._on_complete: Callable[[], None] | None = None
        self._running = False
        self._after_id: str | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(
        self,
        events: Iterator[AlgorithmEvent],
        on_event: Callable[[AlgorithmEvent], None],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Start animating events.

        Args:
            events: Iterator of AlgorithmEvent (from a solver or generator).
            on_event: Called for each event (typically renderer.handle_event).
            on_complete: Called when all events are consumed.
        """
        self.stop()
        self._event_iter = events
        self._on_event = on_event
        self._on_complete = on_complete
        self._running = True
        self._schedule_next()

    def _schedule_next(self) -> None:
        if self._running:
            self._after_id = self._root.after(self.delay_ms, self._step)

    def _step(self) -> None:
        if not self._running or self._event_iter is None:
            return
        try:
            event = next(self._event_iter)
            if self._on_event is not None:
                self._on_event(event)
            self._schedule_next()
        except StopIteration:
            self._running = False
            self._event_iter = None
            if self._on_complete is not None:
                self._on_complete()

    def stop(self) -> None:
        """Stop the animation."""
        self._running = False
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None
        self._event_iter = None

    def set_speed(self, delay_ms: int) -> None:
        """Update animation speed. Takes effect on next frame."""
        self.delay_ms = max(1, delay_ms)

    def run_instant(
        self,
        events: Iterator[AlgorithmEvent],
        on_event: Callable[[AlgorithmEvent], None],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Run all events instantly without animation."""
        self.stop()
        for event in events:
            on_event(event)
        if on_complete is not None:
            on_complete()

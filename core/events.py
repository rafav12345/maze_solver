"""Event system for decoupled communication between algorithms and visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.grid import GridPosition


class EventType(Enum):
    """Types of events that algorithms can emit."""

    # Solver events
    VISIT = auto()
    BACKTRACK = auto()
    PATH_FOUND = auto()
    PATH_NOT_FOUND = auto()

    # Generator events
    WALL_REMOVED = auto()
    CELL_UPDATED = auto()

    # Multi-agent events (Branch 5)
    AGENT_MOVE = auto()
    AGENT_COLLISION = auto()
    AGENT_GOAL_REACHED = auto()

    # Dynamic maze events (Branch 6)
    WALL_ADDED = auto()
    COST_CHANGED = auto()
    MAZE_MUTATED = auto()

    # General
    COMPLETE = auto()


@dataclass
class AlgorithmEvent:
    """An event emitted by a solver or generator during execution.

    The renderer consumes these without knowing which algorithm produced them.
    """

    event_type: EventType
    position: GridPosition
    secondary_position: GridPosition | None = None
    path: list[GridPosition] | None = None
    metadata: dict = field(default_factory=dict)

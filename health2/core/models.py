"""
models.py — Health Voice System Entity Models

Single source of truth for all entity types.
Used for:
  - Tool schema generation (structure.py, mention.py)
  - Output validation (schema_enforcer.py)
  - Type hints throughout the codebase

Do NOT add business logic here. Models are pure data definitions.
"""

from __future__ import annotations
from typing import Annotated, Literal, Optional, Union, List
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────
# MENTION MODELS (Stage 1 output)
# ─────────────────────────────────────────

TemporalEvidence = Literal[
    "explicit_today",
    "active_regimen",
    "future_plan",
    "consultation_relay",
    "unclear"
]

CandidateType = Literal[
    "intake",
    "symptom",
    "activity",
    "machine",
    "device",
    "measurement",
    "meal",
    "outcome",
    "theory",
    "outside",
    "context",
    "other"
]

class Mention(BaseModel):
    raw_mention: str
    candidate_type: CandidateType
    confidence: Literal["high", "low"]
    context: str
    temporal_evidence: TemporalEvidence
    reasoning: str  # chain of thought — why this was extracted and classified this way

class MentionOutput(BaseModel):
    mentions: List[Mention]


# ─────────────────────────────────────────
# ENTITY MODELS (Stage 2 output)
# ─────────────────────────────────────────

IntakeAction = Literal["took", "did_not_take"]
IntakeCategory = Literal["supplement", "prescription", "OTC", "food"]
ActivityStatus = Literal["completed", "planned", "incomplete"]
MachineStatus = Literal["used", "planned"]
InterventionStatus = Literal["active", "completed", "unknown"]
OutcomeDirection = Literal["positive", "negative", "mixed", "unknown"]
TestStatus = Literal["planned", "done"]


class IntakeEntity(BaseModel):
    type: Literal["intake"]
    label: str
    action: IntakeAction
    dose: Optional[float] = None
    unit: Optional[str] = None
    time: Optional[str] = None
    event_date: Optional[str] = None  # for past events: "yesterday", "Friday", "last week"
    category: IntakeCategory
    notes: Optional[str] = None


SymptomStatus = Literal["present", "absent"]

class SymptomEntity(BaseModel):
    type: Literal["symptom"]
    label: str
    status: SymptomStatus = "present"
    onset_time: Optional[str] = None
    severity: Optional[str] = None
    qualifier: Optional[str] = None
    duration: Optional[str] = None
    interval: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


class ActivityEntity(BaseModel):
    type: Literal["activity"]
    label: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[str] = None
    status: ActivityStatus
    event_date: Optional[str] = None  # for past events: "yesterday", "Friday", "last week"
    notes: Optional[str] = None


class MachineEntity(BaseModel):
    type: Literal["machine"]
    label: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[str] = None
    status: MachineStatus
    event_date: Optional[str] = None  # for past events: "yesterday", "Friday", "last week"
    notes: Optional[str] = None


class DeviceEntity(BaseModel):
    type: Literal["device"]
    label: str
    start_time: Optional[str] = None
    status: Literal["used"]


class MeasurementEntity(BaseModel):
    type: Literal["measurement"]
    metric: str
    value: Optional[float] = None  # null if no numeric value stated — use notes for directional observations
    unit: Optional[str] = None
    time: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None   # e.g. "down after 2 AM", "elevated", "in the cellar"

    @field_validator("value", mode="before")
    @classmethod
    def value_must_be_numeric_or_none(cls, v):
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("measurement value cannot be boolean")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"measurement value must be a number or null, got: {v!r}")


class MealEntity(BaseModel):
    type: Literal["meal"]
    label: str
    time: Optional[str] = None
    eaten_out: Optional[bool] = None
    restaurant: Optional[str] = None  # only if eaten_out is True and restaurant name is mentioned
    description: Optional[str] = None  # free-text handoff for downstream food system — preserve as spoken


class InterventionEntity(BaseModel):
    type: Literal["intervention"]
    label: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: InterventionStatus
    notes: Optional[str] = None


class OutcomeEntity(BaseModel):
    type: Literal["outcome"]
    linked_to: str
    what: Optional[str] = None        # what changed: "language recognition", "caffeine sensitivity", "nocturia"
    onset_time: Optional[str] = None
    qualifier: Optional[str] = None
    direction: OutcomeDirection


class TestEntity(BaseModel):
    type: Literal["test"]
    label: str
    time: Optional[str] = None
    status: TestStatus
    result: Optional[str] = None
    notes: Optional[str] = None


class ContextEntity(BaseModel):
    type: Literal["context"]
    raw_text: str
    related_to: Optional[str] = None


class TheoryEntity(BaseModel):
    type: Literal["theory"]
    raw_text: str
    linked_to_label: Optional[str] = None
    linked_to_type: Optional[str] = None


OutsideSubtype = Literal["device_failure", "source_discrepancy", "procurement", "operational", "general"]

class OutsideEntity(BaseModel):
    type: Literal["outside"]
    raw_text: str
    subtype: Optional[OutsideSubtype] = None  # device_failure: unreliable source/sensor issue
    notes: Optional[str] = None  # additional context, related recommendations, or follow-up


# ─────────────────────────────────────────
# UNION TYPE
# ─────────────────────────────────────────

HealthEntity = Annotated[
    Union[
        IntakeEntity,
        SymptomEntity,
        ActivityEntity,
        MachineEntity,
        DeviceEntity,
        MeasurementEntity,
        MealEntity,
        InterventionEntity,
        OutcomeEntity,
        TestEntity,
        ContextEntity,
        TheoryEntity,
        OutsideEntity,
    ],
    Field(discriminator="type")
]


class EntityOutput(BaseModel):
    entities: List[HealthEntity] = Field(
        description="List of structured health entities extracted from the transcript"
    )


# ─────────────────────────────────────────
# TOOL SCHEMA GENERATION
# ─────────────────────────────────────────

def get_entity_tool_schema() -> dict:
    """
    Generate Anthropic tool input_schema from EntityOutput model.
    Used by structure.py to enforce structured output via tool_use.
    """
    return EntityOutput.model_json_schema()


def get_mention_tool_schema() -> dict:
    """
    Generate Anthropic tool input_schema from MentionOutput model.
    Used by mention.py to enforce structured output via tool_use.
    """
    return MentionOutput.model_json_schema()


# ─────────────────────────────────────────
# ENTITY SCHEMAS DICT (backward compat)
# Legacy format used by validate.py prompt injection.
# Keep until validate.py is migrated to tool_use.
# ─────────────────────────────────────────

ENTITY_SCHEMAS = {
    "intake": {
        "fields": ["label", "action", "dose", "unit", "time", "event_date", "category", "notes"],
        "actions": ["took", "did_not_take"],
        "categories": ["supplement", "prescription", "OTC", "food"]
    },
    "symptom": {
        "fields": ["label", "status", "onset_time", "severity", "qualifier", "duration", "interval", "source", "notes"],
        "statuses": ["present", "absent"]
    },
    "activity": {
        "fields": ["label", "start_time", "end_time", "duration", "status", "event_date", "notes"],
        "statuses": ["completed", "planned", "incomplete"]
    },
    "machine": {
        "fields": ["label", "start_time", "end_time", "duration", "status", "event_date", "notes"],
        "statuses": ["used", "planned"]
    },
    "device": {
        "fields": ["label", "start_time", "status"],
        "statuses": ["used"]
    },
    "measurement": {
        "fields": ["metric", "value", "unit", "time", "source"],
        "note": "Numeric values only."
    },
    "meal": {
        "fields": ["label", "time", "eaten_out", "restaurant", "description"]
    },
    "intervention": {
        "fields": ["label", "start_date", "end_date", "status", "notes"],
        "statuses": ["active", "completed", "unknown"]
    },
    "outcome": {
        "fields": ["linked_to", "what", "onset_time", "qualifier", "direction"],
        "directions": ["positive", "negative", "mixed", "unknown"]
    },
    "test": {
        "fields": ["label", "time", "status", "result", "notes"],
        "statuses": ["planned", "done"]
    },
    "context": {
        "fields": ["raw_text", "related_to"]
    },
    "theory": {
        "fields": ["raw_text", "linked_to_label", "linked_to_type"]
    },
    "outside": {
        "fields": ["raw_text", "subtype", "notes"],
        "subtypes": ["device_failure", "source_discrepancy", "procurement", "operational", "general"]
    }
}
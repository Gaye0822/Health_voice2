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
    "explicit_past",      # happened on a specific past day (yesterday, last Friday, etc.)
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
    "intervention",
    "theory",
    "outside",
    "context",
    "other"
]

class Mention(BaseModel):
    raw_mention: str
    inferred_as: Optional[str] = None  # original transcript text if LLM normalized raw_mention
    candidate_type: CandidateType
    confidence: Literal["high", "low"]
    context: str
    temporal_evidence: TemporalEvidence
    event_date_label: Optional[str] = None  # filled only when temporal_evidence = "explicit_past"; user own words: "yesterday", "Friday", "last week"
    reasoning: str  # chain of thought — why this was extracted and classified this way

class MentionOutput(BaseModel):
    mentions: List[Mention]


# ─────────────────────────────────────────
# ENTITY MODELS (Stage 2 output)
# ─────────────────────────────────────────

IntakeAction = Literal["took", "did_not_take"]
IntakeCategory = Literal["supplement", "prescription", "OTC", "food"]
ActivityStatus = Literal["completed", "incomplete", "did_not_complete"]
MachineStatus = Literal["used", "did_not_use"]
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
    event_date: Optional[str] = None  # "yesterday", specific date if intake was not today
    category: IntakeCategory
    is_intervention_dose: Optional[bool] = None  # True if this intake is part of a multi-day protocol
    day_of_protocol: Optional[int] = None        # calculated by schema_enforcer; which day of the protocol this intake belongs to
    notes: Optional[str] = None


class SymptomEntity(BaseModel):
    type: Literal["symptom"]
    label: str
    status: Literal["present", "absent"] = "present"
    onset_time: Optional[str] = None
    severity: Optional[str] = None
    qualifier: Optional[str] = None
    duration: Optional[str] = None
    interval: Optional[str] = None   # retrospective absence duration ("5-6 days", "about a week")
    source: Optional[str] = None     # "retrospective summary" if interval filled from retrospective
    event_date: Optional[str] = None  # "yesterday", specific date if symptom was not today
    notes: Optional[str] = None


class ActivityEntity(BaseModel):
    type: Literal["activity"]
    label: str
    start_time: Optional[str] = None
    duration: Optional[str] = None
    status: ActivityStatus
    event_date: Optional[str] = None
    notes: Optional[str] = None


class MachineEntity(BaseModel):
    type: Literal["machine"]
    label: str
    start_time: Optional[str] = None
    duration: Optional[str] = None
    status: MachineStatus
    event_date: Optional[str] = None
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
    event_date: Optional[str] = None  # "yesterday", "2025-11-14", etc. if meal was not today
    description: Optional[str] = None  # free-text food description handoff for downstream food system


class InterventionEntity(BaseModel):
    type: Literal["intervention"]
    label: str
    start_date: Optional[str] = None   # LLM writes user's words ("two days ago", "last Monday"); schema_enforcer converts to date
    end_date: Optional[str] = None
    status: InterventionStatus = "active"  # default: active — most interventions are ongoing when reported
    duration_days: Optional[int] = None   # total protocol length in days ("10 days" → 10); filled by LLM if stated
    day_of_protocol: Optional[int] = None  # calculated by schema_enforcer; which day of the protocol is today
    notes: Optional[str] = None


class OutcomeEntity(BaseModel):
    type: Literal["outcome"]
    linked_to: str
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
    subtype: OutsideSubtype = "general"
    notes: Optional[str] = None


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
        "fields": ["label", "action", "dose", "unit", "time", "category", "is_intervention_dose", "day_of_protocol", "notes"],
        "actions": ["took", "did_not_take"],
        "categories": ["supplement", "prescription", "OTC", "food"]
    },
    "symptom": {
        "fields": ["label", "onset_time", "severity", "qualifier", "duration"]
    },
    "activity": {
        "fields": ["label", "start_time", "duration", "status", "notes"],
        "statuses": ["completed", "incomplete", "did_not_complete"]
    },
    "machine": {
        "fields": ["label", "start_time", "duration", "status", "notes"],
        "statuses": ["used", "did_not_use"]
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
        "fields": ["label", "time", "eaten_out", "items"]
    },
    "intervention": {
        "fields": ["label", "start_date", "end_date", "status", "notes"],
        "statuses": ["active", "completed", "unknown"]
    },
    "outcome": {
        "fields": ["linked_to", "onset_time", "qualifier", "direction"],
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
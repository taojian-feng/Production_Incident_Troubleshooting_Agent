"""Reasoning engine: evidence -> structured, cited, validated triage steps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from incident_agent.core.errors import LLMOutputError
from incident_agent.domain.enums import TaskType
from incident_agent.domain.models import LLMTriageOutput
from incident_agent.llm.base import LLMRequest, Message
from incident_agent.reasoning.citations import validate_steps
from incident_agent.reasoning.context import render_evidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from incident_agent.domain.models import Alert, Evidence, TriageStep
    from incident_agent.llm.base import LLMProvider
    from incident_agent.llm.router import CostLedger, ModelRouter
    from incident_agent.reasoning.grounding import GroundingChecker

SYSTEM_PROMPT: Final = """\
You are a production incident triage assistant for a regulated FinTech platform.
Production engineers act on your output during live incidents, so every statement
must be verifiable.

Work only from the numbered evidence provided. Produce steps in this order:
1. summary - what is failing and the strongest signal in the logs.
2. precedent - the most similar past incident(s) and how they were resolved.
3. recommendation - concrete, ordered triage actions (one step per action, at most 3).

Cite the evidence ids (for example "E2") that support each step in its "citations"
list. If the evidence does not support a step, say so plainly in that step and leave
its citations empty rather than guessing; such steps are routed to a human reviewer.
Mention ticket ids and error codes exactly as they appear in the evidence. Automated
checks flag any step that names a ticket id or error code without citing an evidence item
containing it, so cite that item, even when you mention the id only to compare or rule it
out. Do not recommend an action unless a cited item describes it or it is a diagnostic
check that follows directly from the cited facts."""

_OUTPUT_SCHEMA: Final = LLMTriageOutput.model_json_schema()


def build_user_prompt(alert: Alert, evidence: Sequence[Evidence]) -> str:
    """The user turn: alert facts, then evidence lines."""
    return (
        f"ALERT {alert.alert_id} [{alert.severity}] service={alert.service}\n"
        f"Title: {alert.title}\nDetails: {alert.description or '-'}\n"
        f"Triggered at: {alert.triggered_at.isoformat()}\n\n"
        f"EVIDENCE\n{render_evidence(evidence)}"
    )


@dataclass(slots=True)
class ReasoningEngine:
    """Calls the routed model with a JSON schema, parses, validates citations."""

    llm: LLMProvider
    router: ModelRouter
    max_tokens: int = 4096
    grounding: GroundingChecker | None = None

    async def triage(
        self, alert: Alert, evidence: Sequence[Evidence], ledger: CostLedger
    ) -> list[TriageStep]:
        """Return validated steps. Unparseable model output -> one repair attempt."""
        model = self.router.route(TaskType.REASON, alert.severity)
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            messages=(Message("user", build_user_prompt(alert, evidence)),),
            max_tokens=self.max_tokens,
            json_schema=_OUTPUT_SCHEMA,
        )
        response = await self.llm.complete(model, request)
        ledger.record(TaskType.REASON, model, response.usage)
        try:
            parsed = LLMTriageOutput.model_validate_json(response.text)
        except ValidationError as first_error:
            repair = LLMRequest(
                system=SYSTEM_PROMPT,
                messages=(
                    *request.messages,
                    Message("assistant", response.text),
                    Message(
                        "user",
                        "That output did not match the schema: "
                        f"{json.dumps(first_error.errors(include_url=False), default=str)[:1500]}. "
                        "Return only the corrected JSON object.",
                    ),
                ),
                max_tokens=self.max_tokens,
                json_schema=_OUTPUT_SCHEMA,
            )
            retry = await self.llm.complete(model, repair)
            ledger.record(TaskType.REASON, model, retry.usage)
            try:
                parsed = LLMTriageOutput.model_validate_json(retry.text)
            except ValidationError as exc:
                msg = "model output failed schema validation twice"
                raise LLMOutputError(msg) from exc
        steps = validate_steps(parsed.steps, evidence, alert.text)
        if self.grounding is None:
            return steps
        return await self.grounding.check(alert, steps, evidence, ledger)

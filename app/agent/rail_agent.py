"""
RailAgent — AI-powered advisory agent for RailMind AI.

Backend
-------
Uses the Groq SDK (free tier) with llama3-8b-8192.
Set GROQ_API_KEY in .env to enable live mode.
All methods degrade gracefully to curated fallbacks when the key is
absent or when the API call fails, so the platform always returns a
valid response.

Get a free key at : https://console.groq.com/
Install           : pip install groq
Model options     : "llama3-8b-8192" | "llama3-70b-8192" | "mixtral-8x7b-32768"
"""


import json
import logging
import os
from dotenv import load_dotenv
from typing import Generator, Optional

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "llama-3.1-8b-instant"


class RailAgent:
    """
    Thin wrapper around the Groq Chat Completions API.

    Public methods
    --------------
    generate_delay_suggestion(delay_data, all_trains) -> dict
        Structured JSON advisory for a delayed train + alternative suggestions.

    generate_safety_alert(safety_result, location) -> str
        Formal, operations-grade safety alert string.

    generate_crowd_advisory(station, current_crowd, alert) -> dict
        Station crowd-management advisory.

    stream_agent_reasoning(delay_data) -> Generator[str, None, None]
        Streaming version for the SSE /delay/agent-stream endpoint.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY", "")
        self._live_mode: bool = bool(api_key)
        self._client = None

        if self._live_mode:
            try:
                from groq import Groq
                self._client = Groq(api_key=api_key)
                logger.info("RailAgent: Groq client initialised (model=%s).", _MODEL)
            except Exception as exc:
                logger.warning(
                    "RailAgent: could not init Groq client (%s) — falling back to mock.", exc
                )
                self._live_mode = False
        else:
            logger.info("RailAgent: GROQ_API_KEY not set — running in mock mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_delay_suggestion(self, delay_data: dict, all_trains: list) -> dict:
        """
        Given a delay-prediction result dict and the full train catalogue,
        ask the LLM to reason step-by-step and return a structured JSON advisory.

        Returns
        -------
        dict with keys:
            agent_message, reasoning, suggested_alternatives,
            urgency ("low" | "medium" | "high"), action_needed
        """
        if self._live_mode:
            return self._live_delay_suggestion(delay_data, all_trains)
        return self._fallback_suggestion(delay_data)

    def generate_safety_alert(
        self, safety_result: dict, location: str = "Unknown"
    ) -> str:
        """
        Generate a formal operations-grade safety alert string.

        Returns
        -------
        str — plain text alert, prefixed with "TRACK SAFETY ALERT:"
              when severity is high or critical.
        """
        if self._live_mode:
            return self._live_safety_alert(safety_result, location)
        return self._fallback_safety_alert(safety_result)

    def generate_crowd_advisory(
        self,
        station: str,
        current_crowd: int,
        alert: Optional[str] = None,
    ) -> dict:
        """
        Generate an operational crowd-management advisory for station staff.

        Returns
        -------
        dict with keys: advisory (str), source (str), model (str | None)
        """
        if self._live_mode:
            return self._live_crowd_advisory(station, current_crowd, alert)
        return self._mock_crowd_advisory(station, current_crowd)

    def stream_agent_reasoning(self, delay_data: dict) -> Generator[str, None, None]:
        """
        Streaming version — yields raw text chunks as they arrive from Groq.
        Designed for the SSE /delay/agent-stream endpoint.

        Usage
        -----
        for chunk in rail_agent.stream_agent_reasoning(delay_data):
            yield f"data: {chunk}\\n\\n"
        """
        if not self._live_mode:
            yield (
                f"Analyzing your train {delay_data.get('train_number', '')}... "
                f"A delay of {delay_data.get('predicted_delay_minutes', 0):.0f} minutes "
                "has been detected. Please check the station display board and NTES app "
                "for the latest updates. Contact Railway enquiry (139) if you need assistance."
            )
            return

        prompt = (
            "You are RailMind AI, an intelligent Indian Railways assistant.\n\n"
            f"Analyzing delay prediction:\n"
            f"Train: {delay_data.get('train_number', 'Unknown')}\n"
            f"Route: {delay_data.get('route', 'Unknown')}\n"
            f"Predicted delay: {delay_data.get('predicted_delay_minutes', 0)} minutes\n"
            f"Category: {delay_data.get('delay_category', 'unknown')}\n\n"
            "Think out loud as you analyze this situation for the passenger.\n"
            "Be conversational and helpful. 3-5 sentences maximum.\n"
            'Start with "Analyzing your train..."'
        )
        try:
            stream = self._client.chat.completions.create(
                model=_MODEL,
                max_tokens=300,
                stream=True,
                messages=[{"role": "user", "content": prompt}],
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("RailAgent stream failed: %s", exc)
            yield f"Agent analysis temporarily unavailable: {exc}"

    # ------------------------------------------------------------------
    # Core Groq helper
    # ------------------------------------------------------------------

    def _call_groq(self, system: str, user: str, max_tokens: int = 600) -> str:
        """Send a system + user message to Groq and return the response text."""
        response = self._client.chat.completions.create(
            model=_MODEL,
            max_tokens=max_tokens,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Live Groq implementations
    # ------------------------------------------------------------------

    def _live_delay_suggestion(self, delay_data: dict, all_trains: list) -> dict:
        system = (
            "You are an Indian Railways passenger assistant AI. "
            "You respond ONLY with valid JSON — no markdown, no explanation, no code blocks."
        )
        user = (
            "A passenger's train has a delay prediction:\n"
            f"- Train: {delay_data.get('train_number', 'Unknown')}\n"
            f"- Route: {delay_data.get('route', 'Unknown')}\n"
            f"- Predicted delay: {delay_data.get('predicted_delay_minutes', 0)} minutes\n"
            f"- Category: {delay_data.get('delay_category', 'unknown')}\n\n"
            f"Available trains in the system: {json.dumps(all_trains[:8], indent=2)}\n\n"
            "Think step by step:\n"
            "1. How severe is this delay for the passenger?\n"
            "2. Are there any trains in the list going to a similar destination?\n"
            "3. What is the best advice for this passenger?\n"
            "4. What should they do RIGHT NOW?\n\n"
            "Respond ONLY with a valid JSON object in this exact format:\n"
            "{\n"
            '  "agent_message": "friendly 2-sentence message to the passenger",\n'
            '  "reasoning": "your step-by-step thinking in 3-4 sentences",\n'
            '  "suggested_alternatives": [\n'
            '    {"train_name": "...", "train_number": "...", "note": "why this is a good option"}\n'
            "  ],\n"
            '  "urgency": "low",\n'
            '  "action_needed": "what the passenger should do right now in one sentence"\n'
            "}"
        )
        try:
            text = self._call_groq(system, user, max_tokens=600)
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as exc:
            logger.error("RailAgent live delay suggestion failed: %s", exc)
            return self._fallback_suggestion(delay_data)

    def _live_safety_alert(self, safety_result: dict, location: str) -> str:
        system = (
            "You are an Indian Railways safety operations AI. "
            "Write formal maintenance alerts using official railway operations language. "
            "Return ONLY the alert text — no markdown, no preamble."
        )
        user = (
            "Track inspection result:\n"
            f"- Defect: {safety_result.get('defect_type', 'unknown')}\n"
            f"- Severity: {safety_result.get('severity', 'unknown')}\n"
            f"- Safe to operate: {safety_result.get('safe_to_operate', False)}\n"
            f"- Location: {location}\n\n"
            "Write a concise maintenance alert in 2-3 sentences. "
            'Start with "TRACK SAFETY ALERT:" if severity is high or critical.'
        )
        try:
            return self._call_groq(system, user, max_tokens=150)
        except Exception as exc:
            logger.error("RailAgent live safety alert failed: %s", exc)
            return self._fallback_safety_alert(safety_result)

    def _live_crowd_advisory(
        self, station: str, crowd: int, alert: Optional[str]
    ) -> dict:
        system = (
            "You are RailMind AI, an operations assistant for Indian Railways station managers. "
            "Give a clear, actionable crowd-management advisory in 3 sentences or fewer. "
            "No bullet points or markdown."
        )
        alert_clause = f" Current alert: {alert}" if alert else ""
        user = (
            f"Station: {station}. Estimated crowd right now: {crowd} passengers.{alert_clause} "
            "What should the station manager do immediately?"
        )
        try:
            advisory = self._call_groq(system, user, max_tokens=200)
            return {"advisory": advisory, "source": "groq", "model": _MODEL}
        except Exception as exc:
            logger.error("RailAgent live crowd advisory failed: %s", exc)
            return self._mock_crowd_advisory(station, crowd)

    # ------------------------------------------------------------------
    # Fallback / mock implementations
    # ------------------------------------------------------------------

    def _fallback_suggestion(self, delay_data: dict) -> dict:
        delay = delay_data.get("predicted_delay_minutes", 0)
        train = delay_data.get("train_number", "your train")
        if delay < 15:
            msg = f"Train {train} is running slightly late but should arrive soon."
            urgency = "low"
        elif delay < 45:
            msg = (
                f"Train {train} is moderately delayed by ~{delay:.0f} minutes. "
                "Consider checking alternate options on NTES."
            )
            urgency = "medium"
        else:
            msg = (
                f"Train {train} is significantly delayed by ~{delay:.0f} minutes. "
                "We strongly recommend exploring alternatives or contacting enquiry (139)."
            )
            urgency = "high"
        return {
            "agent_message": msg,
            "reasoning": "Based on delay duration and category — live AI analysis unavailable.",
            "suggested_alternatives": [],
            "urgency": urgency,
            "action_needed": "Check the station display board and NTES app for live updates.",
        }

    def _fallback_safety_alert(self, safety_result: dict) -> str:
        severity = safety_result.get("severity", "none")
        defect   = safety_result.get("defect_type", "unknown")
        readable = defect.replace("_", " ").title()
        if severity == "critical":
            return (
                f"TRACK SAFETY ALERT: Critical {readable} detected. "
                "Halt all train operations on this segment immediately "
                "pending emergency inspection."
            )
        if severity == "high":
            return (
                f"TRACK SAFETY ALERT: {readable} detected. "
                "Schedule urgent inspection and reduce train speeds on this segment."
            )
        return (
            f"Track inspection complete. {readable} noted. "
            "Routine maintenance recommended at next scheduled window."
        )

    def _mock_crowd_advisory(self, station: str, crowd: int) -> dict:
        if crowd < 300:
            advisory = f"{station} is comfortably occupied. All gates operational."
        elif crowd < 700:
            advisory = (
                f"{station} has moderate footfall. Gate 2 and Gate 4 are recommended "
                "for faster entry."
            )
        else:
            advisory = (
                f"⚠ {station} is experiencing high crowd levels ({crowd} passengers). "
                "Deploy additional RPF personnel, open the overflow waiting area, "
                "and activate PA announcements every 10 minutes."
            )
        return {"advisory": advisory, "source": "mock", "model": None}


# Module-level singleton — import this everywhere.
rail_agent = RailAgent()
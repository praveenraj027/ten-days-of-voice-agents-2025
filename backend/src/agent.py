# agent_sdr.py
import logging
import os
import json
import datetime
from typing import Optional, List, Dict, Any, Literal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    inference,
    cli,
    metrics,
    tokenize,
    room_io,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "shared-data")
FAQ_PATH = os.path.join(DATA_DIR, "company_faq.json")
LEADS_DIR = os.path.join(BASE_DIR, "leads")
os.makedirs(LEADS_DIR, exist_ok=True)

# -- Load company FAQ content (small in-memory RAG) -- #
COMPANY_CONTENT: Dict[str, Any] = {}
FAQ_LIST: List[Dict[str, str]] = []

def _safe_company_field(field: str, default: str = "Unknown") -> str:
    """
    Safely return COMPANY_CONTENT['company'][field] if possible.
    """
    try:
        if isinstance(COMPANY_CONTENT, dict):
            company = COMPANY_CONTENT.get("company", {})
            if isinstance(company, dict):
                return company.get(field, default)
    except Exception as e:
        logger.warning("Error reading company content: %s", e)
    return default

def load_company_content():
    """
    Load and normalize shared-data/company_faq.json into the expected schema:
    {
      "company": {"name": "...", "short_description": "..."},
      "faqs": [ {"id":..., "q":..., "a":...}, ... ],
      "pricing_summary": [...]
    }
    """
    global COMPANY_CONTENT, FAQ_LIST
    if not os.path.exists(FAQ_PATH):
        raise FileNotFoundError(f"company_faq.json not found at {FAQ_PATH}. Create one from the provided JSON.")

    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Start with a safe empty shape
    COMPANY_CONTENT = {"company": {"name": "Unknown", "short_description": ""}}
    COMPANY_CONTENT["faqs"] = []
    COMPANY_CONTENT["pricing_summary"] = []

    if isinstance(raw, dict):
        # If already in expected schema (company is dict)
        if "company" in raw and isinstance(raw["company"], dict):
            # Copy through but ensure keys exist
            COMPANY_CONTENT.update(raw)
            if "faqs" not in COMPANY_CONTENT:
                COMPANY_CONTENT["faqs"] = []
            if "pricing_summary" not in COMPANY_CONTENT:
                COMPANY_CONTENT["pricing_summary"] = raw.get("pricing_summary", [])
        else:
            # Legacy flat schema: company as string + description, faq (list of dicts), pricing dict
            company_name = raw.get("company") if isinstance(raw.get("company"), str) else raw.get("company", "Unknown")
            company_desc = raw.get("description") or raw.get("short_description") or ""
            COMPANY_CONTENT["company"]["name"] = company_name
            COMPANY_CONTENT["company"]["short_description"] = company_desc

            # FAQs: prefer 'faqs' but also accept 'faq'
            faqs_raw = raw.get("faqs") or raw.get("faq") or raw.get("faq_list") or []
            normalized_faqs = []
            for i, item in enumerate(faqs_raw):
                if isinstance(item, dict):
                    q = item.get("q") or item.get("question") or item.get("question_text") or item.get("question") or ""
                    a = item.get("a") or item.get("answer") or item.get("answer_text") or item.get("answer") or ""
                else:
                    q = f"faq_{i}"
                    a = str(item)
                normalized_faqs.append({"id": f"faq_{i}", "q": q, "a": a})
            COMPANY_CONTENT["faqs"] = normalized_faqs

            # Pricing: convert to simple list of summaries if present
            pricing_raw = raw.get("pricing") or {}
            pricing_summary = []
            if isinstance(pricing_raw, dict):
                for product, note in pricing_raw.items():
                    pricing_summary.append({"product": product, "note": str(note)})
            elif isinstance(pricing_raw, list):
                pricing_summary = pricing_raw
            COMPANY_CONTENT["pricing_summary"] = pricing_summary

            # Meta: keep other keys tucked away
            meta_keys = {k: v for k, v in raw.items() if k not in ("company", "description", "short_description", "faq", "faqs", "pricing", "pricing_summary")}
            if meta_keys:
                COMPANY_CONTENT["meta"] = meta_keys
    else:
        logger.warning("Loaded company_faq.json but it's not a dict; got %s", type(raw))
        # leave default COMPANY_CONTENT

    # Final fallbacks
    if "faqs" not in COMPANY_CONTENT:
        COMPANY_CONTENT["faqs"] = []
    if "pricing_summary" not in COMPANY_CONTENT:
        COMPANY_CONTENT["pricing_summary"] = []

    FAQ_LIST = COMPANY_CONTENT.get("faqs", [])
    logger.info("Loaded company content. Company name=%s, #faqs=%d, #pricing_items=%d",
                _safe_company_field("name"), len(FAQ_LIST), len(COMPANY_CONTENT.get("pricing_summary", [])))

# attempt to load at import time (keep previous behavior)
try:
    load_company_content()
except Exception as e:
    logger.exception("Failed loading company content: %s", e)
    COMPANY_CONTENT = {"company": {"name": "Unknown", "short_description": ""}, "faqs": [], "pricing_summary": []}
    FAQ_LIST = []

# ---------- TTS voices (reuse Murf config you had) ---------- #
TTS_SDR = murf.TTS(
    voice="en-US-matthew",
    style="Conversation",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True,
)

# ---------- Helpers: FAQ search & lead state ---------- #

def faq_lookup(query: str) -> Optional[Dict[str, str]]:
    """
    Very simple keyword match: return the first FAQ where any keyword appears in question or answer.
    """
    q = query.lower()
    for faq in FAQ_LIST:
        if q in faq.get("q", "").lower() or q in faq.get("a", "").lower():
            return faq
    # try keyword match of words
    words = [w for w in q.split() if len(w) > 3]
    for faq in FAQ_LIST:
        text = (faq.get("q", "") + " " + faq.get("a", "")).lower()
        if any(w in text for w in words):
            return faq
    return None

def _ensure_lead_state(session) -> Dict[str, Any]:
    ud = session.userdata
    lead = ud.get("lead")
    if not isinstance(lead, dict):
        lead = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None
        }
        ud["lead"] = lead
    return lead

def _save_lead_to_file(lead: Dict[str, Any]) -> str:
    # make filename safe
    name_part = (lead.get("email") or lead.get("name") or "lead").replace(" ", "_").replace("@", "_at_")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(LEADS_DIR, f"{ts}_{name_part}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"lead": lead, "company": COMPANY_CONTENT.get("company")}, f, indent=2)
    return path

# ---------- SDR Agent ---------- #

# ---------- SDR Agent (patched to proactively collect leads) ---------- #

class SDRAgent(Agent):
    """
    Voice SDR agent for a single product/company.
    Tools:
      - answer_faq: answer product/company/pricing questions from loaded FAQ
      - collect_lead: store lead fields as they are provided
      - save_lead: persist lead to disk and return path

    Behavior notes:
      - Actively ask for lead details when the caller shows interest.
      - Use the collect_lead tool to record each field as soon as the user provides it.
      - When all required fields are captured OR the user says "I'm done"/"Thanks",
        call save_lead to persist and return a final summary.
    """

    REQUIRED_LEAD_FIELDS = ["name", "company", "email"]  # minimal required to save
    OPTIONAL_LEAD_FIELDS = ["role", "use_case", "team_size", "timeline"]
    ALL_LEAD_FIELDS = REQUIRED_LEAD_FIELDS + OPTIONAL_LEAD_FIELDS

    def __init__(self, *, tts=None, extra_instructions: str = "", **kwargs):
        company_name = _safe_company_field("name", "Unknown")
        company_short = _safe_company_field("short_description", "")

        # Clear, prescriptive instructions so the LLM will call the tools in a deterministic flow.
        base_instructions = f"""
You are an SDR (sales development rep) voice agent for this company:

Company: {company_name}
Short description: {company_short}

Goals:
1) Understand the caller's needs by asking 'What brought you here today?' and a short follow-up.
2) If the caller shows interest in evaluating the product or requests pricing/demo, OFFER to capture a few quick details to pass to Sales.
3) Collect lead fields in this natural order: name, company, email, role, use_case, team_size, timeline.
   - Ask one question at a time.
   - After the user answers, call the tool `collect_lead(field, value)` with the field name and the user's value.
   - If the user replies with multiple fields at once (e.g. "I'm Alice from Acme, email alice@acme.com"), call `collect_lead` for each parsed field.
4) Once all REQUIRED fields (name OR email) are present, you may ask whether they'd like a demo or schedule a meeting.
5) If the user says "that's all", "thanks", "I’m done", or you have all required fields, call the tool `save_lead()` and then read a short verbal summary:
   - Example summary: "Thanks — I saved Alice (Acme), alice@acme.com. Use case: CRM evaluation for a 10-person team, timeline: soon. I'll pass this to sales."
6) ALWAYS answer product/pricing/company questions from the loaded FAQ using the tool `answer_faq(question)` and never invent precise pricing or limits not present in FAQ. If FAQ lacks the detail, say you don't have that info and offer to connect to sales or schedule a demo.

Tool usage rules (be explicit and consistent):
- To answer product questions: call `answer_faq(question)` and return that tool output to the caller.
- To store lead info: call `collect_lead(field, value)` where field is one of: {', '.join(self.ALL_LEAD_FIELDS)}.
- To persist: call `save_lead()` when the user is done or required fields are collected.

Dialog examples (use these styles, but keep it short and conversational):
- Greeting: "Hi — I'm an SDR for {company_name}. What brought you here today?"
- Offer to capture details: "If you'd like, I can quickly take a few details so our sales team can follow up — can I get your name?"
- After saved: "Thanks — I've saved [name] ([company]), [email]. I'll pass this to our sales team."

{extra_instructions}
"""
        super().__init__(instructions=base_instructions, tts=tts or TTS_SDR, **kwargs)

    # ----------------- Tools ----------------- #

    @function_tool()
    async def answer_faq(self, context: RunContext, question: str) -> str:
        """Return answer from our FAQ list, or a safe fallback if not found."""
        faq = faq_lookup(question)
        if faq:
            return faq["a"]
        short = _safe_company_field("short_description", "")
        if short and any(w in question.lower() for w in ["what do you do", "what is", "about your product", "what does your product do"]):
            return short
        return "I don't have that information in my FAQ. I can book a demo or take your details and connect you with a sales rep."

    @function_tool()
    async def collect_lead(self, context: RunContext, field: str, value: str) -> str:
        """
        Save one lead field. field should be one of the keys in lead state.
        """
        session = context.session
        lead = _ensure_lead_state(session)
        if field not in lead:
            return f"Unknown field '{field}'. Valid fields: {', '.join(lead.keys())}"
        lead[field] = value
        return f"Noted: {field} set."

    @function_tool()
    async def save_lead(self, context: RunContext) -> str:
        """
        Save the lead JSON to disk and return the file path.
        """
        session = context.session
        lead = _ensure_lead_state(session)
        # require at least name or email
        if not (lead.get("email") or lead.get("name")):
            return "I need at least a name or email to save the lead. Could you provide one?"
        path = _save_lead_to_file(lead)
        return f"Saved lead to {path}"

    # ----------------- Conversation entry ----------------- #
    async def on_enter(self) -> None:
        # Greet and ask first question to start the flow.
        await self.session.generate_reply(
            instructions=(
                "Greet the caller warmly as the SDR for the company and ask: "
                "'Hi — I'm an SDR for {company_name}. What brought you here today, and what are you working on?'\n\n"
                "If the user expresses interest in evaluating the product or asks pricing/demo, say: "
                "'Great — I can quickly capture a few details so our sales team can follow up. May I have your name?' "
                "Then WAIT for user response. When the user answers with name, call the `collect_lead` tool."
            )
        )


# ---------- Entrypoint and session setup ---------- #

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=TTS_SDR,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    session.userdata = {}
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    ctx.add_shutdown_callback(log_usage)
    # start SDR
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

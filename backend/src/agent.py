# agent_sdr.py
import logging
import os
import json
import datetime
import sqlite3
import inspect
from typing import Optional, List, Dict, Any

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

# ---------- Fraud DB config ---------- #
FRAUD_DB_PATH = os.path.join(DATA_DIR, "fraud_cases.db")
os.makedirs(DATA_DIR, exist_ok=True)

# -- Load company FAQ content (small in-memory RAG) -- #
COMPANY_CONTENT: Dict[str, Any] = {}
FAQ_LIST: List[Dict[str, str]] = []

def _safe_company_field(field: str, default: str = "Unknown") -> str:
    try:
        if isinstance(COMPANY_CONTENT, dict):
            company = COMPANY_CONTENT.get("company", {})
            if isinstance(company, dict):
                return company.get(field, default)
    except Exception as e:
        logger.warning("Error reading company content: %s", e)
    return default

def load_company_content():
    global COMPANY_CONTENT, FAQ_LIST
    if not os.path.exists(FAQ_PATH):
        # no FAQ, keep defaults
        COMPANY_CONTENT = {"company": {"name": "Unknown", "short_description": ""}, "faqs": [], "pricing_summary": []}
        FAQ_LIST = []
        logger.info("No company_faq.json found at %s; continuing with defaults", FAQ_PATH)
        return

    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    COMPANY_CONTENT = {"company": {"name": "Unknown", "short_description": ""}}
    COMPANY_CONTENT["faqs"] = []
    COMPANY_CONTENT["pricing_summary"] = []

    if isinstance(raw, dict):
        if "company" in raw and isinstance(raw["company"], dict):
            COMPANY_CONTENT.update(raw)
            COMPANY_CONTENT["faqs"] = COMPANY_CONTENT.get("faqs", [])
            COMPANY_CONTENT["pricing_summary"] = COMPANY_CONTENT.get("pricing_summary", [])
        else:
            company_name = raw.get("company") if isinstance(raw.get("company"), str) else raw.get("company", "Unknown")
            company_desc = raw.get("description") or raw.get("short_description") or ""
            COMPANY_CONTENT["company"]["name"] = company_name
            COMPANY_CONTENT["company"]["short_description"] = company_desc
            faqs_raw = raw.get("faqs") or raw.get("faq") or raw.get("faq_list") or []
            normalized_faqs = []
            for i, item in enumerate(faqs_raw):
                if isinstance(item, dict):
                    q = item.get("q") or item.get("question") or item.get("question_text") or ""
                    a = item.get("a") or item.get("answer") or item.get("answer_text") or ""
                else:
                    q = f"faq_{i}"
                    a = str(item)
                normalized_faqs.append({"id": f"faq_{i}", "q": q, "a": a})
            COMPANY_CONTENT["faqs"] = normalized_faqs
            pricing_raw = raw.get("pricing") or {}
            pricing_summary = []
            if isinstance(pricing_raw, dict):
                for product, note in pricing_raw.items():
                    pricing_summary.append({"product": product, "note": str(note)})
            elif isinstance(pricing_raw, list):
                pricing_summary = pricing_raw
            COMPANY_CONTENT["pricing_summary"] = pricing_summary
            meta_keys = {k: v for k, v in raw.items() if k not in ("company", "description", "short_description", "faq", "faqs", "pricing", "pricing_summary")}
            if meta_keys:
                COMPANY_CONTENT["meta"] = meta_keys
    else:
        logger.warning("Loaded company_faq.json but it's not a dict; got %s", type(raw))

    FAQ_LIST = COMPANY_CONTENT.get("faqs", [])
    logger.info("Loaded company content. Company name=%s, #faqs=%d", _safe_company_field("name"), len(FAQ_LIST))

# attempt to load at import time (keep previous behavior)
try:
    load_company_content()
except Exception as e:
    logger.exception("Failed loading company content: %s", e)
    COMPANY_CONTENT = {"company": {"name": "Unknown", "short_description": ""}, "faqs": [], "pricing_summary": []}
    FAQ_LIST = []

# ---------- TTS voices (reuse Murf config you had) ---------- #
def make_murf_tts():
    # create a fresh Murf TTS instance when called inside an event loop/task
    return murf.TTS(
        voice="en-US-matthew",
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
        # If plugin accepts connection options, you can add them here.
        # e.g. connection_options={"timeout": 30, "pool_size": 2}
    )


# ---------- Helpers: FAQ search & lead state (kept for compatibility) ---------- #
def faq_lookup(query: str) -> Optional[Dict[str, str]]:
    q = query.lower()
    for faq in FAQ_LIST:
        if q in faq.get("q", "").lower() or q in faq.get("a", "").lower():
            return faq
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
    name_part = (lead.get("email") or lead.get("name") or "lead").replace(" ", "_").replace("@", "_at_")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(LEADS_DIR, f"{ts}_{name_part}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"lead": lead, "company": COMPANY_CONTENT.get("company")}, f, indent=2)
    return path

# ---------- Simple fraud-case SQLite DB utilities ---------- #
def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(FRAUD_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_fraud_db():
    conn = _connect_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fraud_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT,
        security_identifier TEXT,
        masked_card TEXT,
        transaction_amount TEXT,
        merchant_name TEXT,
        location TEXT,
        timestamp TEXT,
        security_question TEXT,
        security_answer TEXT,
        status TEXT,
        outcome_note TEXT,
        raw_json TEXT
    )
    """)
    conn.commit()
    conn.close()

def seed_sample_fraud_cases():
    conn = _connect_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) as c FROM fraud_cases")
    if cur.fetchone()["c"] > 0:
        conn.close()
        return
    samples = [
        {
            "user_name": "John",
            "security_identifier": "SID-1001",
            "masked_card": "**** 4242",
            "transaction_amount": "$129.99",
            "merchant_name": "ABC Industry",
            "location": "Bengaluru, India",
            "timestamp": "2025-11-25 14:32:00",
            "security_question": "What is the name of your first pet?",
            "security_answer": "fluffy",
            "status": "pending_review",
        },
        {
            "user_name": "Alice",
            "security_identifier": "SID-1002",
            "masked_card": "**** 9876",
            "transaction_amount": "$599.00",
            "merchant_name": "GadgetWorld",
            "location": "Mumbai, India",
            "timestamp": "2025-11-25 09:15:00",
            "security_question": "In which city were you born?",
            "security_answer": "pune",
            "status": "pending_review",
        },
        {
            "user_name": "Bob",
            "security_identifier": "SID-1003",
            "masked_card": "**** 1111",
            "transaction_amount": "$15.49",
            "merchant_name": "FoodHub",
            "location": "Delhi, India",
            "timestamp": "2025-11-24 20:02:00",
            "security_question": "What was your high school mascot?",
            "security_answer": "tiger",
            "status": "pending_review",
        }
    ]
    for s in samples:
        cur.execute("""
            INSERT INTO fraud_cases
            (user_name, security_identifier, masked_card, transaction_amount, merchant_name,
             location, timestamp, security_question, security_answer, status, outcome_note, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s["user_name"], s["security_identifier"], s["masked_card"], s["transaction_amount"],
            s["merchant_name"], s["location"], s["timestamp"], s["security_question"],
            s["security_answer"], s["status"], "", json.dumps(s)
        ))
    conn.commit()
    conn.close()

def load_case_for_username(username: str) -> Optional[Dict[str, Any]]:
    conn = _connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM fraud_cases
        WHERE user_name = ? AND status = 'pending_review'
        ORDER BY id ASC
        LIMIT 1
    """, (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

def update_fraud_case(case_id: int, status: str, outcome_note: str):
    conn = _connect_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE fraud_cases
        SET status = ?, outcome_note = ?, raw_json = ?
        WHERE id = ?
    """, (status, outcome_note, json.dumps({"last_updated": datetime.datetime.utcnow().isoformat()+"Z", "note": outcome_note}), case_id))
    conn.commit()
    conn.close()

# initialize DB at import time (safe: idempotent)
try:
    init_fraud_db()
    seed_sample_fraud_cases()
except Exception as e:
    logger.exception("Failed initializing fraud DB: %s", e)

# ---------- Fraud Agent ---------- #

class FraudAgent(Agent):
    """
    Fraud alert voice agent.
    Tools exposed:
      - fetch_case(username) -> returns fraud case JSON or a message if none
      - verify_security(case_id, answer) -> returns True/False (and message)
      - confirm_decision(case_id, decision) -> updates DB and returns summary message
    Notes:
      - Uses only fake data seeded in shared-data/fraud_cases.db
      - NEVER ask for or store sensitive data like full card numbers, PINs, passwords, or OTPs.
    """

    def __init__(self, *, tts=None, extra_instructions: str = "", **kwargs):
        company_name = _safe_company_field("name", "Your Bank")
        base_instructions = f"""
You are a calm, professional fraud department representative for {company_name}.
Your goal is to:
 - Identify a pending suspicious transaction by asking for the user's username (first name).
 - Verify identity using a single non-sensitive security question fetched from the fraud case.
 - Read the suspicious transaction details (merchant, amount, masked card, approximate time and location).
 - Ask the user whether they made the transaction ('yes' or 'no').
 - If user confirms -> mark case as 'confirmed_safe' and summarize actions.
 - If user denies -> mark case as 'confirmed_fraud' and explain mock actions (card blocked, dispute opened).
 - If verification fails or user reply is unclear -> mark as 'verification_failed' and end call.

Never request full card numbers, PINs, OTPs, or passwords. Treat all database entries as fake/demo-only.
When interacting with the caller, use short, reassuring sentences.
{extra_instructions}
"""
        # Use only the tts passed in by the caller (do not reference a removed global)
        super().__init__(instructions=base_instructions, tts=tts , **kwargs)

    # --------------------------- Tools --------------------------- #

    @function_tool()
    async def fetch_case(self, context: RunContext, username: str) -> str:
        """
        Returns JSON string of the pending fraud case for username, or a message when none found.
        """
        case = load_case_for_username(username.strip())
        if not case:
            return json.dumps({"found": False, "message": f"No pending suspicious transactions found for {username}."})
        # Remove sensitive fields just in case (we only stored masked card)
        sanitized = {
            "found": True,
            "id": case["id"],
            "user_name": case["user_name"],
            "security_identifier": case["security_identifier"],
            "masked_card": case["masked_card"],
            "transaction_amount": case["transaction_amount"],
            "merchant_name": case["merchant_name"],
            "location": case["location"],
            "timestamp": case["timestamp"],
            "security_question": case["security_question"],
            "status": case["status"],
        }
        return json.dumps(sanitized)

    @function_tool()
    async def verify_security(self, context: RunContext, case_id: int, answer: str) -> str:
        """
        Verify a non-sensitive security question answer for the given case_id.
        Returns a JSON string with verification result and a simple message.
        """
        # load case by id
        conn = _connect_db()
        cur = conn.cursor()
        cur.execute("SELECT security_answer FROM fraud_cases WHERE id = ?", (case_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return json.dumps({"ok": False, "message": "Case not found."})
        expected = (row["security_answer"] or "").strip().lower()
        if expected == answer.strip().lower():
            return json.dumps({"ok": True, "message": "Verification successful."})
        else:
            # update case status to verification_failed
            update_fraud_case(case_id, "verification_failed", "Verification failed during automated call.")
            return json.dumps({"ok": False, "message": "Verification failed."})

    @function_tool()
    async def confirm_decision(self, context: RunContext, case_id: int, decision: str) -> str:
        """
        decision: expected 'yes' or 'no' (case-insensitive)
        Updates DB status and returns a short outcome summary string.
        """
        dec = (decision or "").strip().lower()
        if dec in ("yes", "y"):
            update_fraud_case(case_id, "confirmed_safe", "Customer confirmed transaction as legitimate via automated call.")
            return json.dumps({"status": "confirmed_safe", "message": "Marked transaction as legitimate. No further action."})
        elif dec in ("no", "n"):
            # mock actions (block card, open dispute)
            update_fraud_case(case_id, "confirmed_fraud", "Customer denied transaction; card blocked and dispute initiated (mock).")
            return json.dumps({"status": "confirmed_fraud", "message": "Marked transaction as fraudulent. Card blocked and dispute initiated (mock)."})
        else:
            update_fraud_case(case_id, "verification_failed", "Unclear answer provided during confirmation step.")
            return json.dumps({"status": "verification_failed", "message": "Unclear response; ended call for safety."})

    # --------------------------- Entry dialog --------------------------- #

    async def on_enter(self) -> None:
        # Provide an explicit human-readable flow instruction so model will call the tools in order.
        company_name = _safe_company_field("name", "Your Bank")
        await self.session.generate_reply(
            instructions=(
                "You are a fraud-detection representative for {company_name}. Start by greeting the caller and saying: "
                "'Hello — this is the Fraud Department at {company_name}. For your safety, I will ask one quick verification question.' "
                "Prompt the user: 'Please tell me your username (first name) so I can look up a suspicious transaction.'\n\n"
                "When the user provides the username, call the tool `fetch_case(username)` to retrieve the pending case. "
                "If no case found, read the tool message and politely end the call. "
                "If a case is found, ask the security question from the case and WAIT for answer. "
                "When the user answers, call `verify_security(case_id, answer)`. If verification fails, end the call. "
                "If verification succeeds, read the transaction details (merchant, amount, masked card, approximate time and location), "
                "then ask: 'Did you make this transaction? Please answer yes or no.' Wait for reply and then call `confirm_decision(case_id, decision)`; read the resulting message and end call."
            )
        )

# ---------- Entrypoint and session setup ---------- #

def prewarm(proc: JobProcess):
    # Keep existing VAD prewarm behavior
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    import asyncio
    ctx.log_context_fields = {"room": ctx.room.name}

    # make TTS inside event loop
    tts = make_murf_tts()
    logger.info("Created TTS instance: %s (close method present: %s)", type(tts), hasattr(tts, "close"))

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=tts,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    logger.info("AgentSession created, session.tts is not None: %s", getattr(session, "tts", None) is not None)

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

    # Ensure TTS closed on shutdown
    async def _close_tts():
        try:
            close_coro = getattr(tts, "close", None)
            if close_coro:
                if inspect.iscoroutinefunction(close_coro):
                    await close_coro()
                else:
                    close_coro()
                logger.info("Closed TTS instance cleanly on shutdown.")
        except Exception as e:
            logger.exception("Error closing Murf TTS: %s", e)

    ctx.add_shutdown_callback(_close_tts)

    # start Fraud agent (pass the same tts instance to agent)
    await session.start(
        agent=FraudAgent(tts=tts),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

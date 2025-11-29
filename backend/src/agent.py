# agent_gamemaster_full.py
"""
Voice Game Master Agent — Full implementation for Day 8 with Advanced Goals:
- JSON world state (characters, locations, events, quests)
- Player character sheet & inventory
- Dice roll mechanics (1-20) with modifiers
- Multiple universe presets (Fantasy, Cyberpunk, Space Opera)
- Save & Load game state to shared-data/gamesessions.json

Drop this file into your backend (replace or alongside existing agent file).
Run it the same way you run other agents (cli.run_app WorkerOptions entrypoint_fnc).

This file is designed for the LiveKit agents runtime similar to the ShoppingAgent example.
Adjust imports / plugin usage to match your environment if needed.
"""

import logging
import os
import json
import uuid
import random
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent_gamemaster_full")
load_dotenv(".env.local")

# ------------------- Storage paths -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "shared-data")
os.makedirs(DATA_DIR, exist_ok=True)
GAMES_PATH = os.path.join(DATA_DIR, "gamesessions.json")

# ensure games file
if not os.path.exists(GAMES_PATH):
    with open(GAMES_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)

# ------------------- Universe presets -------------------
UNIVERSE_PRESETS = {
    "fantasy": {
        "name": "Classic Fantasy",
        "system_prompt": "You are a Dungeons & Dragons–style Game Master. Universe: A fantasy world filled with ancient ruins, mystical forests, and lurking monsters. Tone: Dramatic and immersive.",
        "starter_scene": "You awaken in a dim forest clearing beneath a silvered moon. A twig snaps in the brush. A robed figure approaches. What do you do?",
        "default_pc": {"name": "Adventurer", "class": "Ranger", "hp": 12, "max_hp": 12, "str": 12, "int": 10, "luck": 8, "inventory": ["dagger", "torch"]},
        "locations": {"forest_clearing": {"name": "Forest Clearing", "desc": "Tall oaks ring a small patch of moonlit grass.", "paths": ["old_road"]}},
    },
    "cyberpunk": {
        "name": "Neon City",
        "system_prompt": "You are a Game Master running a gritty cyberpunk city—neon, corporations, and netrunners. Tone: Noir and edgy.",
        "starter_scene": "Rain hisses on neon glass. You stand under a flickering sign. A drone hums overhead. What do you do?",
        "default_pc": {"name": "Runner", "class": "Hacker", "hp": 10, "max_hp": 10, "str": 8, "int": 14, "luck": 10, "inventory": ["cyberdeck", "credit_chip"]},
        "locations": {"neon_alley": {"name": "Neon Alley", "desc": "Steam and lights, graffiti and wet asphalt.", "paths": ["main_hub"]}},
    },
    "space": {
        "name": "Space Opera",
        "system_prompt": "You are a Game Master running a space opera—vast starships and alien worlds. Tone: Epic and cinematic.",
        "starter_scene": "The airlock breathes behind you. Stars spread like spilled salt. The captain's voice crackles. What do you do?",
        "default_pc": {"name": "Pilot", "class": "Scout", "hp": 14, "max_hp": 14, "str": 10, "int": 12, "luck": 9, "inventory": ["laser_pistol", "nav_chip"]},
        "locations": {"airlock": {"name": "Ship Airlock", "desc": "Cold metal, a view of the void.", "paths": ["bridge"]}},
    },
}

# ------------------- Helpers -------------------

def _load_games():
    with open(GAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_games(games):
    with open(GAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2, ensure_ascii=False)


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

# Dice roll mechanic

def roll_d20(modifier: int = 0) -> dict:
    roll = random.randint(1, 20)
    total = roll + modifier
    outcome = "failure"
    if roll == 20:
        outcome = "critical_success"
    elif roll == 1:
        outcome = "critical_failure"
    elif total >= 20:
        outcome = "success"
    elif total >= 10:
        outcome = "partial_success"
    else:
        outcome = "failure"
    return {"roll": roll, "modifier": modifier, "total": total, "outcome": outcome}

# Simple TTS chunking same approach as earlier
async def speak_text(session, tts, text: str):
    if not text:
        return
    text = text.strip()
    if not text:
        return
    max_chars = 700
    chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    for chunk in chunks:
        try:
            if hasattr(tts, "stream_text"):
                async for _ in tts.stream_text(chunk):
                    pass
            elif hasattr(tts, "synthesize"):
                maybe = tts.synthesize(chunk)
                if inspect.isawaitable(maybe):
                    await maybe
        except Exception:
            # best-effort; don't crash the agent on TTS failures
            logger.exception("TTS chunk failed")
            break

# ------------------- GameMaster Agent -------------------
class GameMasterAgent(Agent):
    def __init__(self, *, tts=None, universe: str = "fantasy"):
        # choose preset
        preset = UNIVERSE_PRESETS.get(universe, UNIVERSE_PRESETS['fantasy'])
        system_prompt = preset['system_prompt'] + "\nRules: Always end your message with 'What do you do?'. Use the session world_state when deciding outcomes."
        super().__init__(instructions=system_prompt, tts=tts)
        self.universe_key = universe

    async def on_enter(self) -> None:
        # initialize session world state (stored in session.userdata['world_state'])
        ud = getattr(self.session, 'userdata', None)
        if ud is None:
            self.session.userdata = {}
            ud = self.session.userdata
        if 'world_state' not in ud:
            ud['world_state'] = self._fresh_world_state(self.universe_key)
        intro = ud['world_state']['scene']['description']
        # speak starter scene
        await speak_text(self.session, self.session.tts, intro)

    def _fresh_world_state(self, universe_key: str) -> dict:
        preset = UNIVERSE_PRESETS.get(universe_key, UNIVERSE_PRESETS['fantasy'])
        world = {
            'id': str(uuid.uuid4()),
            'universe': universe_key,
            'created_at': _now_iso(),
            'updated_at': _now_iso(),
            'scene': {'id': 'start', 'description': preset['starter_scene'], 'location': list(preset.get('locations', {}).keys())[0]},
            'player': preset['default_pc'].copy(),
            'npcs': {},
            'locations': preset.get('locations', {}).copy(),
            'events': [],
            'quests': {'active': [], 'completed': []},
            'history': [],
        }
        return world

    # ------------------- Utility function tools -------------------
    # Expose some programmatic helpers to the LLM (if function calling is supported)
    from livekit.agents import function_tool

    @function_tool()
    async def get_world_state(self, context: RunContext) -> str:
        ud = context.session.userdata
        ws = ud.get('world_state') if ud else None
        return json.dumps({'ok': True, 'world_state': ws})

    @function_tool()
    async def save_game(self, context: RunContext, save_name: str = "autosave") -> str:
        ud = context.session.userdata
        ws = ud.get('world_state') if ud else None
        if not ws:
            return json.dumps({'ok': False, 'message': 'No active world to save.'})
        games = _load_games()
        games[save_name] = {'saved_at': _now_iso(), 'world_state': ws}
        _save_games(games)
        return json.dumps({'ok': True, 'message': f'Saved as {save_name}.'})

    @function_tool()
    async def load_game(self, context: RunContext, save_name: str) -> str:
        games = _load_games()
        if save_name not in games:
            return json.dumps({'ok': False, 'message': f'Save {save_name} not found.'})
        ws = games[save_name]['world_state']
        context.session.userdata['world_state'] = ws
        return json.dumps({'ok': True, 'message': f'Loaded {save_name}.', 'world_state': ws})

    @function_tool()
    async def list_saves(self, context: RunContext) -> str:
        games = _load_games()
        keys = list(games.keys())
        return json.dumps({'ok': True, 'saves': keys})

    @function_tool()
    async def roll(self, context: RunContext, modifier: int = 0) -> str:
        r = roll_d20(int(modifier))
        # optionally update history
        ud = context.session.userdata
        ws = ud.get('world_state')
        entry = {'time': _now_iso(), 'type': 'roll', 'roll': r}
        if ws:
            ws['events'].append(entry)
            ws['updated_at'] = _now_iso()
        return json.dumps({'ok': True, 'result': r})

    @function_tool()
    async def show_inventory(self, context: RunContext) -> str:
        ud = context.session.userdata
        ws = ud.get('world_state')
        inv = ws['player'].get('inventory', []) if ws else []
        return json.dumps({'ok': True, 'inventory': inv})

    @function_tool()
    async def show_stats(self, context: RunContext) -> str:
        ud = context.session.userdata
        ws = ud.get('world_state')
        if not ws:
            return json.dumps({'ok': False, 'message': 'No character found.'})
        player = ws['player']
        stats = {k: v for k, v in player.items() if k in ('name','class','hp','max_hp','str','int','luck')}
        return json.dumps({'ok': True, 'stats': stats})

    @function_tool()
    async def switch_universe(self, context: RunContext, universe_key: str) -> str:
        if universe_key not in UNIVERSE_PRESETS:
            return json.dumps({'ok': False, 'message': f'Unknown universe {universe_key}. Options: {list(UNIVERSE_PRESETS.keys())}'})
        ud = context.session.userdata
        ud['world_state'] = self._fresh_world_state(universe_key)
        return json.dumps({'ok': True, 'message': f'Switched to {universe_key}.', 'world_state': ud['world_state']})

    # ------------------- High-level LLM prompt helper -------------------
    # The agent runtime will call LLM based on the system instructions. We include
    # a small helper to append the player action into world_state.history so that
    # the GM can rely on it for continuity.
    async def append_player_action(self, player_text: str):
        ud = getattr(self.session, 'userdata', None)
        if ud is None:
            self.session.userdata = {}
            ud = self.session.userdata
        ws = ud.get('world_state')
        if not ws:
            # initialize with default universe
            ud['world_state'] = self._fresh_world_state(self.universe_key)
            ws = ud['world_state']
        ws['history'].append({'time': _now_iso(), 'player': player_text})
        ws['updated_at'] = _now_iso()

    # This method shows how you can process the player's text and then speak a response.
    # The LiveKit agent runtime will already pass player messages to the LLM with
    # system instructions; however, depending on your runtime you might want to
    # intercept messages for commands (save/load/roll) before letting the LLM run.
    async def on_message(self, message_text: str) -> None:
        # called when a message arrives (runtime-dependent). We implement a best-effort
        # handler that recognizes explicit commands and otherwise lets the LLM drive the narrative.
        text = (message_text or "").strip().lower()
        # quick command parsing
        if text.startswith("save ") or text == "save":
            name = text.split(maxsplit=1)[1] if " " in text else "autosave"
            await self.save_game(RunContext(session=self.session), save_name=name)
            reply = f"Saved your game as {name}. What do you do?"
            await speak_text(self.session, self.session.tts, reply)
            return
        if text.startswith("load "):
            name = text.split(maxsplit=1)[1]
            res = json.loads(await self.load_game(RunContext(session=self.session), save_name=name))
            if res.get('ok'):
                reply = f"Loaded {name}. " + self.session.userdata['world_state']['scene']['description']
            else:
                reply = res.get('message', 'Could not load.')
            await speak_text(self.session, self.session.tts, reply)
            return
        if text.startswith("roll"):
            # allow "roll" or "roll +2"
            parts = text.split()
            modifier = 0
            if len(parts) >= 2:
                try:
                    modifier = int(parts[1].lstrip('+-'))
                except Exception:
                    modifier = 0
            r = roll_d20(modifier)
            # store event
            ud = self.session.userdata
            ws = ud.get('world_state')
            ws['events'].append({'time': _now_iso(), 'type': 'roll', 'result': r})
            await speak_text(self.session, self.session.tts, f"You rolled {r['roll']} + {r['modifier']} = {r['total']}. Outcome: {r['outcome']}. What do you do?")
            return
        if text in ("inventory", "what's in my bag", "what do i have"):
            inv = self.session.userdata['world_state']['player'].get('inventory', [])
            await speak_text(self.session, self.session.tts, "You have: " + (", ".join(inv) or "nothing") + ". What do you do?")
            return

        # otherwise, treat as normal player action: append history and ask the LLM to produce next scene
        await self.append_player_action(message_text)
        # The LLM will be called by framework using system prompt + conversation. We provide a short bridging
        # speak while the LLM generates (optional)
        await speak_text(self.session, self.session.tts, "...thinking...")
        # Note: In many runtimes you do not call the LLM manually here. The agent framework will use your
        # system prompt and conversation history to generate the next assistant message and speak it.

# ------------------- prewarm & entrypoint -------------------

def prewarm(proc: JobProcess):
    try:
        proc.userdata['vad'] = silero.VAD.load()
    except Exception:
        logger.exception('Failed to prewarm VAD')


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    tts = murf.TTS(voice="en-US-matthew", style="Narration")
    logger.info("Created Murf TTS instance for GameMasterAgent.")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=tts,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get('vad'),
        preemptive_generation=True,
    )

    session.userdata = {}

    async def _close_tts():
        try:
            close_coro = getattr(tts, "close", None)
            if close_coro:
                if inspect.iscoroutinefunction(close_coro):
                    await close_coro()
                else:
                    close_coro()
                logger.info("Closed Murf TTS instance cleanly on shutdown.")
        except Exception as e:
            logger.exception("Error closing Murf TTS: %s", e)

    ctx.add_shutdown_callback(_close_tts)

    # start session with GameMasterAgent (choose 'fantasy' by default)
    await session.start(
        agent=GameMasterAgent(tts=tts, universe='fantasy'),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

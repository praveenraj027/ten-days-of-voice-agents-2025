import logging
import os
import json
import datetime
from typing import Optional, List

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
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly coffee shop barista for a brand called Falcon Brew.

You take exactly ONE coffee order at a time and maintain this JSON order state:

{
  "drinkType": "string",
  "size": "string",
  "milk": "string",
  "extras": ["string"],
  "name": "string"
}

Your workflow:
- Greet the customer.
- Ask what they'd like to drink.
- Ask clarifying questions until ALL fields in the order are filled.
- You can accept combined phrases (e.g. "a large iced latte with oat milk") and then only ask for what is still missing.
- Never guess any detail. If you are unsure, ASK.
- As you learn anything about the order, call the `update_order` tool with the fields you understood.
- When you believe the order is complete, FIRST:
  - Read back a short summary to the customer in one or two sentences.
  - THEN call the `save_order` tool to persist the order.
- Use the `reset_order` tool if the customer wants to completely change their order.
- Keep responses short and conversational, like a real barista talking at the counter.
""",
        )

    @function_tool()
    async def update_order(
        self,
        context: RunContext,
        drinkType: str = "",
        size: str = "",
        milk: str = "",
        extras: Optional[List[str]] = None,
        name: str = "",
    ) -> str:
        """
        Update the current coffee order.

        Call this whenever the customer provides ANY order details
        (drinkType, size, milk, extras, or name).

        You can call this multiple times as you collect more information.
        Only send the fields that changed.
        """
        # initialize order state if missing
        userdata = context.session.userdata
        order = userdata.get("order")
        if order is None:
            order = {
                "drinkType": "",
                "size": "",
                "milk": "",
                "extras": [],
                "name": "",
            }
            userdata["order"] = order

        if drinkType:
            order["drinkType"] = drinkType
        if size:
            order["size"] = size
        if milk:
            order["milk"] = milk
        if extras is not None:
            order["extras"] = extras
        if name:
            order["name"] = name

        missing = [
            key
            for key in ("drinkType", "size", "milk", "name")
            if not order.get(key)
        ]

        return json.dumps(
            {
                "order": order,
                "missing_fields": missing,
            }
        )


    @function_tool()
    async def reset_order(self, context: RunContext) -> str:
        """
        Clear the current order and start over.

        Use this if the customer wants to change their whole order.
        """
        context.session.userdata["order"] = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": "",
        }
        return "Order has been reset. Start a fresh order with the customer."

    @function_tool()
    async def save_order(self, context: RunContext) -> str:
        """
        Save the current completed order to a JSON file.

        Only call this AFTER all fields are filled and you've
        verbally confirmed the order with the customer.
        """
        order = context.session.userdata.get("order")

        if not order:
            return "There is no active order to save."

        missing = [
            key
            for key in ("drinkType", "size", "milk", "name")
            if not order.get(key)
        ]

        if missing:
            return (
                "Order is not complete yet. Missing fields: "
                + ", ".join(missing)
            )

        # Make sure extras is at least an empty list
        if order.get("extras") is None:
            order["extras"] = []

        # Ensure orders directory exists
        os.makedirs("orders", exist_ok=True)

        # Use UTC timestamp to avoid clashes
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        safe_name = (order.get("name") or "guest").replace(" ", "_")
        filename = f"orders/order-{timestamp}-{safe_name}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2)

        context.session.userdata["last_saved_order_file"] = filename

        # Short human-readable summary (this is what your frontend can show as text)
        summary = (
            f"Order for {order['name']}: "
            f"{order['size']} {order['drinkType']} with {order['milk']} milk"
        )
        if order["extras"]:
            summary += f", extras: {', '.join(order['extras'])}."

        return summary



    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        
    )
    session.userdata = {}

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

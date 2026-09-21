"""
SanataniBot - A respectful Sanatana Dharma assistant for Discord.

Stack : discord.py + Google Gemini (gemini-2.5-flash)
Deploy: Render.com free Web Service (includes a tiny health-check HTTP server)

Required environment variables:
    DISCORD_TOKEN   - Your Discord bot token
    GEMINI_API_KEY  - Your Google AI Studio (Gemini) API key
    PORT            - (Optional) Set automatically by Render; defaults to 8080
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import discord
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("SanataniBot")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

DISCORD_MAX_LEN = 2000        # Discord's hard limit per message
CONTEXT_MAX_CHARS = 1500      # Max chars of the bot's earlier message kept as context
GEMINI_TIMEOUT_SECONDS = 90   # Give up on a stuck Gemini request after this long

ERROR_GENERIC = (
    "Hari Om 🙏 Forgive me, I encountered a difficulty while preparing your answer. "
    "Kindly try again in a little while."
)
ERROR_RATE_LIMIT = (
    "Hari Om 🙏 Many seekers are asking questions at this moment and I must rest briefly. "
    "Kindly ask again after a minute."
)
ERROR_EMPTY = (
    "Hari Om 🙏 I was unable to form a response to that. "
    "Kindly rephrase your question and I shall try again."
)
GREETING = (
    "Hari Om! 🙏 I am SanataniBot. Ask me anything about Sanatana Dharma, "
    "scriptures, Sanskrit, Panchang, or daily sadhana."
)

SYSTEM_INSTRUCTION = """
You are SanataniBot, a respectful, humble and traditionalist Hindu assistant living
inside a Discord server. You are deeply knowledgeable in Sanatana Dharma.

YOUR AREAS OF KNOWLEDGE
- The Shruti and Smriti literature: the four Vedas (Rigveda, Yajurveda, Samaveda,
  Atharvaveda), the Upanishads, the Brahmanas and Aranyakas, the Bhagavad Gita,
  the Ramayana, the Mahabharata, and the Puranas (Bhagavata, Vishnu, Shiva, Devi, etc.).
- The darshanas (Nyaya, Vaisheshika, Samkhya, Yoga, Mimamsa, Vedanta) and the major
  acharyas and sampradayas (Advaita, Vishishtadvaita, Dvaita, Shaiva Siddhanta,
  Vaishnava, Shakta, Smarta and others), presented fairly and without disparaging any.
- Sanskrit language and grammar (vyakarana): sandhi, samasa, vibhakti, dhatu, lakara,
  chandas (metre), pronunciation, and word meanings.
- Panchang concepts: tithi, vara, nakshatra, yoga, karana, masa, paksha, samvatsara,
  ayana, rutu, muhurta, Rahu Kalam, Ekadashi, Sankranti, Amavasya, Purnima and festival
  observances.
- Daily sadhana and traditions: sandhyavandanam, japa, dhyana, puja vidhi, stotras,
  aarti, vrata, upavasa, samskaras, satvik living, and dharmic conduct.

YOUR CHARACTER AND MANNER
- Always be humble, gentle, polite and respectful. Greet warmly (for example
  "Namaste", "Hari Om" or "Jai Shri Ram / Om Namah Shivaya" where fitting) but do not
  overdo it or repeat greetings unnecessarily.
- Ground every answer in authentic Dharmic tradition and cite the source when you
  know it (e.g. "Bhagavad Gita 2.47", "Taittiriya Upanishad 1.11"). Never fabricate
  verses, verse numbers or citations. If you are not certain, say so plainly.
- When you use a Sanskrit term or quote a shloka, give the Devanagari where helpful,
  a clear transliteration (IAST or simple readable Roman), and a simple meaning.
- Where traditions or sampradayas differ, present the differing views respectfully
  rather than declaring one to be the only truth.
- For Panchang, tithi, muhurta or festival-date questions: explain the concepts
  clearly, but be honest that exact timings depend on date, location and the regional
  system followed (e.g. Amanta vs Purnimanta, Drik vs Surya Siddhanta). Never invent
  precise timings; recommend verifying with a trusted local Panchang or a learned
  Pandit for anything important.
- For serious personal, medical, legal or ritual-eligibility questions, offer
  traditional perspective with humility and suggest consulting a qualified guru,
  acharya, Pandit or appropriate professional.
- Never mock, belittle or attack any person, community or faith. If asked about other
  religions, respond respectfully and factually. Avoid political arguments and
  divisive content; steer towards understanding and dharmic values.
- If a question is far outside your scope, politely say so and, where possible, relate
  it back to what you can helpfully offer.

FORMAT
- You are replying on Discord: keep answers clear and reasonably concise, ideally under
  1500 characters unless the user asks for depth. Use simple Discord markdown
  (bold, italics, short bullet lists, > quotes for shlokas). Avoid large tables and
  headings.
""".strip()

# --------------------------------------------------------------------------- #
# Fail fast if secrets are missing
# --------------------------------------------------------------------------- #
if not DISCORD_TOKEN:
    raise SystemExit("Hari Om - DISCORD_TOKEN environment variable is not set.")
if not GEMINI_API_KEY:
    raise SystemExit("Hari Om - GEMINI_API_KEY environment variable is not set.")


# --------------------------------------------------------------------------- #
# Health-check web server (keeps Render's port scan / health check happy)
# --------------------------------------------------------------------------- #
class HealthCheckHandler(BaseHTTPRequestHandler):
    """Minimal handler that answers 200 OK to GET (and HEAD) requests."""

    def _respond(self, include_body: bool) -> None:
        body = b"SanataniBot is running. Hari Om!"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming convention)
        self._respond(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond(include_body=False)

    def log_message(self, format: str, *args) -> None:  # silence per-request logs
        return


def start_health_server() -> ThreadingHTTPServer:
    """
    Bind the port immediately (so a bind failure crashes loudly at startup instead of
    silently inside a thread), then serve requests from a background daemon thread.
    """
    port = int(os.environ.get("PORT", 8080))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, name="health-server", daemon=True)
    thread.start()
    log.info("Health-check server listening on port %s", server.server_address[1])
    return server


# --------------------------------------------------------------------------- #
# Gemini setup
# --------------------------------------------------------------------------- #
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION,
    # Gemini 2.5 "thinking" tokens count against this limit, so keep it generous.
    generation_config={"temperature": 0.7, "max_output_tokens": 8192},
)

# --------------------------------------------------------------------------- #
# Discord setup
# --------------------------------------------------------------------------- #
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# Never let model output ping @everyone / @here / roles.
SAFE_MENTIONS = discord.AllowedMentions(everyone=False, roles=False, users=True, replied_user=True)


def clean_prompt(message: discord.Message, bot_id: int) -> str:
    """
    Build the prompt text from a Discord message:
      * strips the bot's own mention tag (<@ID> / <@!ID>)
      * turns other users' raw mention tags into readable @names
    """
    text = re.sub(rf"<@!?{bot_id}>", "", message.content)
    for user in message.mentions:
        if user.id == bot_id:
            continue
        name = f"@{user.display_name}"
        text = text.replace(f"<@{user.id}>", name).replace(f"<@!{user.id}>", name)
    return text.strip()


def split_message(text: str, limit: int = DISCORD_MAX_LEN) -> list[str]:
    """Split text into chunks <= limit, preferring newline/space boundaries."""
    text = text.strip()
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def extract_text(response) -> str:
    """Safely pull text out of a Gemini response ('' if blocked or empty)."""
    try:
        return (response.text or "").strip()
    except (ValueError, IndexError):
        # .text raises when the response has no usable part (e.g. safety block).
        return ""


async def get_referenced_message(message: discord.Message) -> discord.Message | None:
    """Return the message this one replies to, fetching it only if not already resolved."""
    ref = message.reference
    if ref is None or ref.message_id is None:
        return None

    resolved = ref.resolved
    if isinstance(resolved, discord.Message):
        return resolved
    if isinstance(resolved, discord.DeletedReferencedMessage):
        return None

    try:
        return await message.channel.fetch_message(ref.message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None


async def send_reply(message: discord.Message, content: str) -> None:
    """Reply to the message; if that fails (e.g. original deleted), post in the channel."""
    try:
        await message.reply(content, allowed_mentions=SAFE_MENTIONS)
    except discord.HTTPException:
        await message.channel.send(content, allowed_mentions=SAFE_MENTIONS)


async def send_error(message: discord.Message, text: str) -> None:
    """Best-effort error notification; never raises."""
    try:
        await send_reply(message, text)
    except Exception:  # noqa: BLE001
        log.exception("Could not deliver error message to the channel")


@client.event
async def on_ready() -> None:
    log.info("Logged in as %s (ID: %s)", client.user, client.user.id)


@client.event
async def on_message(message: discord.Message) -> None:
    # Ignore ourselves (and other bots, to avoid bot-to-bot loops).
    if message.author == client.user or message.author.bot:
        return

    is_mentioned = client.user in message.mentions

    # Cheap early exit: not tagged and not a reply -> nothing to do (no API calls).
    if not is_mentioned and message.reference is None:
        return

    referenced = await get_referenced_message(message) if message.reference else None
    is_reply_to_bot = referenced is not None and referenced.author == client.user

    # Trigger ONLY when tagged or when replying directly to one of our messages.
    if not (is_mentioned or is_reply_to_bot):
        return

    prompt = clean_prompt(message, client.user.id)

    if not prompt:
        await send_error(message, GREETING)
        return

    # If the user is replying to us, give Gemini our previous message as context.
    if is_reply_to_bot and referenced.content:
        previous = referenced.content[:CONTEXT_MAX_CHARS]
        full_prompt = (
            "Context - your previous message in this conversation:\n"
            f"{previous}\n\n"
            "The user's reply to it:\n"
            f"{prompt}"
        )
    else:
        full_prompt = prompt

    async with message.channel.typing():
        try:
            response = await asyncio.wait_for(
                model.generate_content_async(full_prompt),
                timeout=GEMINI_TIMEOUT_SECONDS,
            )
            answer = extract_text(response) or ERROR_EMPTY

            chunks = split_message(answer)
            await send_reply(message, chunks[0])
            for chunk in chunks[1:]:
                await message.channel.send(chunk, allowed_mentions=SAFE_MENTIONS)

        except google_exceptions.ResourceExhausted:
            log.warning("Gemini rate limit / quota reached")
            await send_error(message, ERROR_RATE_LIMIT)
        except Exception:  # noqa: BLE001 - deliberately broad so the bot never crashes
            log.exception("Error while generating or sending response")
            await send_error(message, ERROR_GENERIC)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    start_health_server()
    try:
        client.run(DISCORD_TOKEN, log_handler=None)
    except discord.LoginFailure:
        log.critical("Hari Om - Discord rejected DISCORD_TOKEN. Please check the token.")
        raise SystemExit(1)
    except discord.PrivilegedIntentsRequired:
        log.critical(
            "Hari Om - Enable 'Message Content Intent' for the bot in the "
            "Discord Developer Portal (Bot -> Privileged Gateway Intents)."
        )
        raise SystemExit(1)

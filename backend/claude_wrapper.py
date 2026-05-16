"""
claude_wrapper.py — Day-6 natural-language wrapper around the routine planner.

What this does
--------------
Takes (user profile, current routine output from the planner, chat history,
user message) and asks Claude to produce a conversational response.

What this DOES NOT do
---------------------
Make recommendation decisions. Those come from the trained gradient-boosting
ranker (Day 3) and the MILP routine planner (Day 4). Claude's job is to
explain and discuss the routine — it must not invent products, prices,
brands, or compatibility scores. The system prompt enforces this; the
backend should also never expose the full catalog to Claude (only the
already-selected routine + the user's profile).

Model: claude-sonnet-4-6 (user-selected; switch via CLAUDE_MODEL env var)
"""
import os
from anthropic import Anthropic

# Reads ANTHROPIC_API_KEY from environment automatically (loaded by .env in app.py).
_client = Anthropic()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 800

SYSTEM_PROMPT = """You are the conversational layer of the Makeup Mastermind recommender system.

The actual recommendations come from a trained gradient-boosting ranker plus a MILP routine planner. Your job is to describe the routine that's been produced and answer the user's questions about it — not to recommend products yourself.

HARD RULES (do not break these):
1. You may ONLY reference products that appear in ROUTINE_DATA below.
2. You may NOT invent or recommend products from your own knowledge — even if the user asks for alternatives that aren't in ROUTINE_DATA.
3. You may NOT make up brand names, prices, ingredients, or compatibility scores.
4. Always quote prices and brand/product names exactly as they appear in ROUTINE_DATA.
5. If the user asks about a product or category that isn't in ROUTINE_DATA, say you don't have data on it and suggest they regenerate the routine with adjusted preferences.
6. When asked "why was X chosen?", reference the actual compatibility_score and any matched_sensitivities or category from ROUTINE_DATA for that product.

TONE: Warm but factual. Beauty-consultant voice. Cite specific numbers (prices, scores) when justifying choices. Keep responses under ~150 words unless the user asks for detail.

If the user asks to change their profile (skin type, budget, sensitivities, style), acknowledge it but tell them they'll need to update the sidebar and regenerate the routine — you can't re-run the planner yourself."""


def _format_routine_for_claude(routine_dict: dict) -> str:
    """Compact text representation of the planner output Claude can quote from."""
    routine = routine_dict.get("routine") or []
    if not routine:
        return f"No routine generated. Status: {routine_dict.get('status', 'unknown')}."

    lines = []
    for p in routine:
        flags = p.get("matched_sensitivities") or []
        flag_str = f" · flagged: {', '.join(flags)}" if flags else ""
        lines.append(
            f"- [{p.get('category_unified', '?')}] "
            f"{p.get('brand', '?')} {p.get('product_name', '?')} · "
            f"${float(p.get('price_usd', 0)):.2f} · "
            f"compatibility {float(p.get('predicted_score', 0)):.3f}{flag_str}"
        )
    lines.append(f"Total cost: ${float(routine_dict.get('total_cost', 0)):.2f}")
    lines.append(f"Slot coverage: {100 * float(routine_dict.get('slot_coverage', 0)):.0f}%")
    if routine_dict.get("dropped_slots"):
        lines.append(f"Dropped slots (infeasibility): {', '.join(routine_dict['dropped_slots'])}")
    return "\n".join(lines)


def _format_profile(profile: dict) -> str:
    sens = profile.get("sensitivities") or []
    sens_str = ", ".join(sens) if sens else "none"
    return (
        f"skin_type={profile.get('skin_type', '?')} · "
        f"skin_tone={profile.get('skin_tone', '?')} · "
        f"style={profile.get('style', '?')} · "
        f"budget=${profile.get('budget', '?')} · "
        f"sensitivities={sens_str}"
    )


def chat(user_message: str, user_profile: dict, routine_dict: dict, chat_history: list) -> str:
    """
    Return Claude's natural-language reply.

    Parameters
    ----------
    user_message : str
        What the user just typed.
    user_profile : dict
        Profile fields from the survey: skin_type, skin_tone, style, budget, sensitivities.
    routine_dict : dict
        The planner's output (already converted to JSON-friendly form in app.py):
        routine (list of dicts), total_cost, status, slot_coverage, dropped_slots, ...
    chat_history : list[dict]
        Prior turns as [{role: "user"|"assistant", content: "..."}, ...].

    Returns
    -------
    str
        Claude's reply text. Caller wraps it in the API response envelope.
    """
    routine_text = _format_routine_for_claude(routine_dict)
    profile_text = _format_profile(user_profile)

    framed_user = (
        f"USER_PROFILE: {profile_text}\n\n"
        f"ROUTINE_DATA:\n{routine_text}\n\n"
        f"USER_QUESTION: {user_message}"
    )

    # Build the full message list. Prior chat history is preserved; the framed
    # context is only attached to the LATEST user turn, so the model gets fresh
    # routine state every turn without bloating the history.
    #
    # The frontend pushes each user message into chat_history BEFORE calling
    # /api/chat, and also passes it as `user_message`. To avoid sending the
    # current question twice (once raw, once framed), drop the trailing user
    # turn from history when it matches.
    hist = list(chat_history or [])
    if hist and hist[-1].get("role") == "user" and hist[-1].get("content") == user_message:
        hist = hist[:-1]

    messages = []
    for turn in hist:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": framed_user})

    response = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    # Extract text from the first text block (the only block type for a plain chat reply).
    for block in response.content:
        if block.type == "text":
            return block.text
    return "(no response generated)"

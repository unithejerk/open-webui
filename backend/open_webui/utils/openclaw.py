"""
OpenClaw provider helpers.

OpenClaw is a stateful agent backend.  It receives a session key
(x-openclaw-session-key) that uniquely identifies a conversation and stores
the full message history server-side.  Because OpenClaw already knows the
conversation, Open WebUI should only send the system prompt + messages that
have accumulated since the last user turn — not the entire history on every
request.
"""

from open_webui.models.users import UserModel


# ---------------------------------------------------------------------------
# Provider gate
# ---------------------------------------------------------------------------


def is_openclaw_provider(api_config: dict) -> bool:
    """Return True if this connection targets an OpenClaw agent backend."""
    return api_config.get('agents_provider') == 'openclaw'


# ---------------------------------------------------------------------------
# Session key
# ---------------------------------------------------------------------------


def derive_openclaw_session_key(
    metadata: dict | None,
    user: UserModel,
) -> str:
    """
    Derive a stable session key for the x-openclaw-session-key header.

    Format: openwebui:<user_id>:<chat_id>

    When chat_id is unavailable in metadata, falls back to:
        openwebui:<user_id>:default
    """
    user_id = user.id if user else 'anonymous'
    chat_id = (metadata or {}).get('chat_id', 'default')
    return f'openwebui:{user_id}:{chat_id}'


# ---------------------------------------------------------------------------
# Request mutation helpers
# ---------------------------------------------------------------------------


def inject_openclaw_headers(
    headers: dict,
    api_config: dict,
    metadata: dict | None,
    user: UserModel,
) -> None:
    """Mutate *headers* in-place to add x-openclaw-session-key if applicable."""
    if not is_openclaw_provider(api_config):
        return
    if not user:
        return
    headers['x-openclaw-session-key'] = derive_openclaw_session_key(metadata, user)


def inject_openclaw_body(
    payload: dict,
    api_config: dict,
    user: UserModel,
) -> None:
    """
    Mutate *payload* in-place to add the user field for OpenClaw connections.

    Collision rules:
    - If payload['user'] is already a string, preserve it (caller intent wins).
    - If payload['user'] is a dict/object (e.g. from pipeline plumbing), leave
      it alone -- OpenClaw expects a string, but pipeline objects serve a
      different purpose and we don't stomp them.
    - Otherwise, set payload['user'] = str(user.id).
    """
    if not is_openclaw_provider(api_config):
        return
    if not user:
        return
    if 'user' in payload:
        existing = payload['user']
        if isinstance(existing, str):
            return  # caller-supplied string user -- preserve it
        # dict/object or other type -- leave untouched (pipeline plumbing)
        return
    payload['user'] = str(user.id)


def adapt_openclaw_input_images(payload: dict, api_config: dict) -> None:
    """
    Rewrite input_image items from OpenAI flat format to OpenClaw source-wrapper
    format, for OpenClaw connections only.

    OpenAI format:  {'type': 'input_image', 'image_url': 'https://...'}
    OpenClaw format: {'type': 'input_image', 'source': {'type': 'url', 'url': 'https://...'}}

    Also handles base64 data URLs by converting them to source.type='base64'.
    Mutates payload['input'] in-place.
    """
    if not is_openclaw_provider(api_config):
        return
    input_items = payload.get('input')
    if not isinstance(input_items, list):
        return
    for item in input_items:
        if item.get('type') != 'input_image':
            continue
        image_url = item.pop('image_url', None)
        if image_url is None:
            continue
        if isinstance(image_url, str) and image_url.startswith('data:'):
            # base64 data URL
            item['source'] = {'type': 'base64', 'data': image_url}
        else:
            item['source'] = {'type': 'url', 'url': image_url}


# ---------------------------------------------------------------------------
# Message trimming
# ---------------------------------------------------------------------------
#
# OpenClaw stores the full conversation server-side, keyed by the session
# key.  Re-sending the entire history on every turn wastes tokens and defeats
# the purpose of having a stateful backend.  Only the system prompt +
# messages since the most recent user turn need to be forwarded.


def trim_openclaw_chat_messages(payload: dict, api_config: dict) -> None:
    """
    Trim payload['messages'] for OpenClaw connections.

    Keeps only system messages + messages from the most recent user role
    message onward (so tool-call chains stay intact).
    """
    if not is_openclaw_provider(api_config):
        return
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        return

    system_msgs = [m for m in messages if m.get('role') == 'system']
    tail_start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user':
            tail_start = i
            break

    tail = messages[tail_start:]  # empty when no user message found (safe)
    payload['messages'] = system_msgs + tail


def trim_openclaw_responses_input(payload: dict, api_config: dict) -> None:
    """
    Trim payload['input'] for OpenClaw connections.

    Keeps only system instruction items + items from the most recent user
    role message onward.
    """
    if not is_openclaw_provider(api_config):
        return
    input_items = payload.get('input')
    if not isinstance(input_items, list) or not input_items:
        return

    system_items = [
        it for it in input_items
        if it.get('type') == 'message' and it.get('role') == 'system'
    ]
    tail_start = len(input_items)
    for i in range(len(input_items) - 1, -1, -1):
        item = input_items[i]
        if item.get('type') == 'message' and item.get('role') == 'user':
            tail_start = i
            break

    tail = input_items[tail_start:]  # empty when no user message (safe)
    payload['input'] = system_items + tail

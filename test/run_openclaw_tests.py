"""Standalone runner for OpenClaw helper tests (no pytest required)."""

# Copy the helpers inline (avoid module-level import side effects from
# open_webui.utils.openclaw, which pulls in UserModel / FastAPI config).
# Keep in sync with open_webui/utils/openclaw.py.

def is_openclaw_provider(api_config):
    return api_config.get('agents_provider') == 'openclaw'

def derive_openclaw_session_key(metadata, user):
    user_id = user.id if user else 'anonymous'
    chat_id = (metadata or {}).get('chat_id', 'default')
    return f'openwebui:{user_id}:{chat_id}'

def inject_openclaw_headers(headers, api_config, metadata, user):
    if not is_openclaw_provider(api_config):
        return
    if not user:
        return
    headers['x-openclaw-session-key'] = derive_openclaw_session_key(metadata, user)

def inject_openclaw_body(payload, api_config, user):
    if not is_openclaw_provider(api_config):
        return
    if not user:
        return
    if 'user' in payload:
        existing = payload['user']
        if isinstance(existing, str):
            return
        return
    payload['user'] = str(user.id)

def adapt_openclaw_input_images(payload, api_config):
    if api_config.get('agents_provider') != 'openclaw':
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
            item['source'] = {'type': 'base64', 'data': image_url}
        else:
            item['source'] = {'type': 'url', 'url': image_url}

def trim_openclaw_chat_messages(payload, api_config):
    """Trim payload['messages'] for OpenClaw connections."""
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
    tail = messages[tail_start:]
    payload['messages'] = system_msgs + tail

def trim_openclaw_responses_input(payload, api_config):
    """Trim payload['input'] for OpenClaw connections."""
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
    tail = input_items[tail_start:]
    payload['input'] = system_items + tail


import sys
from unittest.mock import MagicMock

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS {name}')
    else:
        failed += 1
        print(f'  FAIL {name}')

oc = {'agents_provider': 'openclaw'}
hermes = {'agents_provider': 'hermes'}
empty = {}
user = MagicMock()
user.id = 'user-abc-123'
user.name = 'Test User'
meta = {'chat_id': 'chat-xyz-789', 'message_id': 'msg-456'}

print('=== is_openclaw_provider ===')
check('matches openclaw', is_openclaw_provider(oc) is True)
check('rejects hermes', is_openclaw_provider(hermes) is False)
check('rejects empty', is_openclaw_provider(empty) is False)

print('\n=== derive_openclaw_session_key ===')
key1 = derive_openclaw_session_key(meta, user)
key2 = derive_openclaw_session_key(meta, user)
check('stable with chat_id', key1 == key2)
check('format', key1 == 'openwebui:user-abc-123:chat-xyz-789')
check('different chat different key',
      derive_openclaw_session_key({'chat_id': 'A'}, user) !=
      derive_openclaw_session_key({'chat_id': 'B'}, user))
check('same user+chat same key',
      derive_openclaw_session_key({'chat_id': 'X'}, user) ==
      derive_openclaw_session_key({'chat_id': 'X'}, user))
check('fallback no chat_id',
      derive_openclaw_session_key({}, user) == 'openwebui:user-abc-123:default')
check('fallback metadata None',
      derive_openclaw_session_key(None, user) == 'openwebui:user-abc-123:default')
check('anonymous user',
      derive_openclaw_session_key({'chat_id': 'c1'}, None) == 'openwebui:anonymous:c1')

u2 = MagicMock(); u2.id = 'user-B'
check('different user different key',
      derive_openclaw_session_key(meta, user) != derive_openclaw_session_key(meta, u2))

print('\n=== inject_openclaw_headers ===')
h = {}
inject_openclaw_headers(h, oc, meta, user)
check('adds header for openclaw', h.get('x-openclaw-session-key') == 'openwebui:user-abc-123:chat-xyz-789')

h = {}
inject_openclaw_headers(h, hermes, meta, user)
check('noop for non-openclaw', 'x-openclaw-session-key' not in h)

h = {}
inject_openclaw_headers(h, empty, meta, user)
check('noop for empty config', 'x-openclaw-session-key' not in h)

h = {}
inject_openclaw_headers(h, oc, meta, None)
check('noop when no user', 'x-openclaw-session-key' not in h)

h = {'Authorization': 'Bearer sk-abc'}
inject_openclaw_headers(h, oc, meta, user)
check('preserves existing headers', h['Authorization'] == 'Bearer sk-abc')
check('adds session key alongside', 'x-openclaw-session-key' in h)

print('\n=== inject_openclaw_body ===')
p = {'model': 'gpt-4', 'messages': []}
inject_openclaw_body(p, oc, user)
check('adds user field', p.get('user') == 'user-abc-123')

p = {'model': 'gpt-4'}
inject_openclaw_body(p, hermes, user)
check('noop for non-openclaw', 'user' not in p)

p = {'model': 'gpt-4'}
inject_openclaw_body(p, empty, user)
check('noop for empty config', 'user' not in p)

p = {'model': 'gpt-4'}
inject_openclaw_body(p, oc, None)
check('noop when no user', 'user' not in p)

p = {'model': 'gpt-4', 'user': 'caller-id'}
inject_openclaw_body(p, oc, user)
check('preserves existing string user', p['user'] == 'caller-id')

p = {'model': 'gpt-4', 'user': {'name': 'pipe', 'id': 'x'}}
inject_openclaw_body(p, oc, user)
check('preserves existing dict user (pipeline)', p['user'] == {'name': 'pipe', 'id': 'x'})

p = {}
inject_openclaw_body(p, oc, user)
check('user is string', isinstance(p['user'], str))

print('\n=== adapt_openclaw_input_images ===')

# Convert URL image
p = {'input': [{'type': 'input_image', 'image_url': 'https://example.com/cat.jpg'}]}
adapt_openclaw_input_images(p, oc)
check('converts image_url to source wrapper',
      p['input'][0] == {'type': 'input_image', 'source': {'type': 'url', 'url': 'https://example.com/cat.jpg'}})
check('image_url key removed', 'image_url' not in p['input'][0])

# Convert base64 data URL
p = {'input': [{'type': 'input_image', 'image_url': 'data:image/jpeg;base64,/9j/4AAQ'}]}
adapt_openclaw_input_images(p, oc)
check('converts data URL to base64 source',
      p['input'][0]['source'] == {'type': 'base64', 'data': 'data:image/jpeg;base64,/9j/4AAQ'})

# Noop for non-openclaw
p = {'input': [{'type': 'input_image', 'image_url': 'https://example.com/cat.jpg'}]}
adapt_openclaw_input_images(p, hermes)
check('noop for hermes', p['input'][0] == {'type': 'input_image', 'image_url': 'https://example.com/cat.jpg'})

# Non-image items untouched
p = {'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}]}
adapt_openclaw_input_images(p, oc)
check('non-image items untouched', p['input'][0]['type'] == 'message')

# No input key
p = {'model': 'gpt-4'}
adapt_openclaw_input_images(p, oc)
check('no input key - no error', 'input' not in p)

# Input is not a list
p = {'input': 'not a list'}
adapt_openclaw_input_images(p, oc)
check('non-list input - no error', p['input'] == 'not a list')

# Already in source format
p = {'input': [{'type': 'input_image', 'source': {'type': 'url', 'url': 'https://x.com'}}]}
adapt_openclaw_input_images(p, oc)
check('already in source format - preserved',
      p['input'][0]['source'] == {'type': 'url', 'url': 'https://x.com'})

print('\n=== trim_openclaw_chat_messages ===')

# Simple turn — keeps only system + last user message
p = {'messages': [
    {'role': 'system', 'content': 'You are helpful.'},
    {'role': 'user', 'content': 'hello'},
    {'role': 'assistant', 'content': 'hi!'},
    {'role': 'user', 'content': 'what can you do?'},
]}
trim_openclaw_chat_messages(p, oc)
check('simple turn: system + last user', len(p['messages']) == 2)
check('simple turn: system preserved', p['messages'][0]['role'] == 'system')
check('simple turn: last user msg', p['messages'][1]['content'] == 'what can you do?')

# Tool call chain — keeps from last user msg
p = {'messages': [
    {'role': 'system', 'content': 'sys'},
    {'role': 'user', 'content': 'weather?'},
    {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'get_weather'}}]},
    {'role': 'tool', 'content': 'sunny, 72F'},
]}
trim_openclaw_chat_messages(p, oc)
check('tool chain: keeps all 4', len(p['messages']) == 4)

# No system — only last user msg
p = {'messages': [
    {'role': 'user', 'content': 'old'},
    {'role': 'assistant', 'content': 'old response'},
    {'role': 'user', 'content': 'new'},
]}
trim_openclaw_chat_messages(p, oc)
check('no system: only last user', p['messages'] == [{'role': 'user', 'content': 'new'}])

# No user message — only system
p = {'messages': [
    {'role': 'system', 'content': 'sys'},
    {'role': 'assistant', 'content': 'hi'},
]}
trim_openclaw_chat_messages(p, oc)
check('no user: only system', p['messages'] == [{'role': 'system', 'content': 'sys'}])

# Empty
p = {'messages': []}
trim_openclaw_chat_messages(p, oc)
check('empty messages', p['messages'] == [])

# Noop for non-openclaw
p = {'messages': [
    {'role': 'system', 'content': 'sys'},
    {'role': 'user', 'content': 'hi'},
]}
orig = list(p['messages'])
trim_openclaw_chat_messages(p, hermes)
check('noop for hermes', p['messages'] == orig)

# No messages key
p = {'model': 'gpt-4'}
trim_openclaw_chat_messages(p, oc)
check('no messages key - no error', 'messages' not in p)

print('\n=== trim_openclaw_responses_input ===')

# Simple turn
p = {'input': [
    {'type': 'message', 'role': 'system', 'content': [{'type': 'output_text', 'text': 'sys'}]},
    {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'old'}]},
    {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'old reply'}]},
    {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'new'}]},
]}
trim_openclaw_responses_input(p, oc)
check('responses: system + last user', len(p['input']) == 2)
check('responses: system preserved', p['input'][0]['role'] == 'system')
check('responses: last user item', p['input'][1]['content'][0]['text'] == 'new')

# Noop for non-openclaw
p = {'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}]}
trim_openclaw_responses_input(p, hermes)
check('responses noop', len(p['input']) == 1)

# Empty input
p = {'input': []}
trim_openclaw_responses_input(p, oc)
check('responses empty', p['input'] == [])

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(0 if failed == 0 else 1)

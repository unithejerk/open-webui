"""Standalone runner for OpenClaw helper tests (no pytest required)."""
import sys
from unittest.mock import MagicMock

# Copy the helpers inline (avoid module-level import side effects)
def _is_openclaw_provider(api_config):
    return api_config.get('agents_provider') == 'openclaw'

def _derive_openclaw_session_key(metadata, user):
    user_id = user.id if user else 'anonymous'
    chat_id = (metadata or {}).get('chat_id', 'default')
    return f'openwebui:{user_id}:{chat_id}'

def _inject_openclaw_headers(headers, api_config, metadata, user):
    if not _is_openclaw_provider(api_config):
        return
    if not user:
        return
    headers['x-openclaw-session-key'] = _derive_openclaw_session_key(metadata, user)

def _inject_openclaw_body(payload, api_config, user):
    if not _is_openclaw_provider(api_config):
        return
    if not user:
        return
    if 'user' in payload:
        existing = payload['user']
        if isinstance(existing, str):
            return
        return
    payload['user'] = str(user.id)


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

print('=== _is_openclaw_provider ===')
check('matches openclaw', _is_openclaw_provider(oc) is True)
check('rejects hermes', _is_openclaw_provider(hermes) is False)
check('rejects empty', _is_openclaw_provider(empty) is False)

print('\n=== _derive_openclaw_session_key ===')
key1 = _derive_openclaw_session_key(meta, user)
key2 = _derive_openclaw_session_key(meta, user)
check('stable with chat_id', key1 == key2)
check('format', key1 == 'openwebui:user-abc-123:chat-xyz-789')
check('different chat different key',
      _derive_openclaw_session_key({'chat_id': 'A'}, user) !=
      _derive_openclaw_session_key({'chat_id': 'B'}, user))
check('same user+chat same key',
      _derive_openclaw_session_key({'chat_id': 'X'}, user) ==
      _derive_openclaw_session_key({'chat_id': 'X'}, user))
check('fallback no chat_id',
      _derive_openclaw_session_key({}, user) == 'openwebui:user-abc-123:default')
check('fallback metadata None',
      _derive_openclaw_session_key(None, user) == 'openwebui:user-abc-123:default')
check('anonymous user',
      _derive_openclaw_session_key({'chat_id': 'c1'}, None) == 'openwebui:anonymous:c1')

# Different user, different key
u2 = MagicMock(); u2.id = 'user-B'
check('different user different key',
      _derive_openclaw_session_key(meta, user) != _derive_openclaw_session_key(meta, u2))

print('\n=== _inject_openclaw_headers ===')
h = {}
_inject_openclaw_headers(h, oc, meta, user)
check('adds header for openclaw', h.get('x-openclaw-session-key') == 'openwebui:user-abc-123:chat-xyz-789')

h = {}
_inject_openclaw_headers(h, hermes, meta, user)
check('noop for non-openclaw', 'x-openclaw-session-key' not in h)

h = {}
_inject_openclaw_headers(h, empty, meta, user)
check('noop for empty config', 'x-openclaw-session-key' not in h)

h = {}
_inject_openclaw_headers(h, oc, meta, None)
check('noop when no user', 'x-openclaw-session-key' not in h)

h = {'Authorization': 'Bearer sk-abc'}
_inject_openclaw_headers(h, oc, meta, user)
check('preserves existing headers', h['Authorization'] == 'Bearer sk-abc')
check('adds session key alongside', 'x-openclaw-session-key' in h)

print('\n=== _inject_openclaw_body ===')
p = {'model': 'gpt-4', 'messages': []}
_inject_openclaw_body(p, oc, user)
check('adds user field', p.get('user') == 'user-abc-123')

p = {'model': 'gpt-4'}
_inject_openclaw_body(p, hermes, user)
check('noop for non-openclaw', 'user' not in p)

p = {'model': 'gpt-4'}
_inject_openclaw_body(p, empty, user)
check('noop for empty config', 'user' not in p)

p = {'model': 'gpt-4'}
_inject_openclaw_body(p, oc, None)
check('noop when no user', 'user' not in p)

p = {'model': 'gpt-4', 'user': 'caller-id'}
_inject_openclaw_body(p, oc, user)
check('preserves existing string user', p['user'] == 'caller-id')

p = {'model': 'gpt-4', 'user': {'name': 'pipe', 'id': 'x'}}
_inject_openclaw_body(p, oc, user)
check('preserves existing dict user (pipeline)', p['user'] == {'name': 'pipe', 'id': 'x'})

p = {}
_inject_openclaw_body(p, oc, user)
check('user is string', isinstance(p['user'], str))

print(f'\n=== RESULTS: {passed} passed, {failed} failed ===')
sys.exit(0 if failed == 0 else 1)

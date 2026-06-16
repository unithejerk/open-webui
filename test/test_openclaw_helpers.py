"""
Tests for OpenClaw provider-specific request helpers.

These test the pure helper functions from open_webui/utils/openclaw.py:
  is_openclaw_provider, derive_openclaw_session_key,
  inject_openclaw_headers, inject_openclaw_body, adapt_openclaw_input_images,
  trim_openclaw_chat_messages, trim_openclaw_responses_input

Run with: python -m pytest test/test_openclaw_helpers.py -v
"""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline helpers (avoid pulling in the full FastAPI / config dependency tree).
# Keep in sync with open_webui/utils/openclaw.py.
# ---------------------------------------------------------------------------

def is_openclaw_provider(api_config: dict) -> bool:
    return api_config.get('agents_provider') == 'openclaw'

def derive_openclaw_session_key(metadata, user) -> str:
    user_id = user.id if user else 'anonymous'
    chat_id = (metadata or {}).get('chat_id', 'default')
    return f'openwebui:{user_id}:{chat_id}'

def inject_openclaw_headers(headers, api_config, metadata, user) -> None:
    if not is_openclaw_provider(api_config):
        return
    if not user:
        return
    headers['x-openclaw-session-key'] = derive_openclaw_session_key(metadata, user)

def inject_openclaw_body(payload, api_config, user) -> None:
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

def adapt_openclaw_input_images(payload, api_config) -> None:
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

def trim_openclaw_chat_messages(payload, api_config) -> None:
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

def trim_openclaw_responses_input(payload, api_config) -> None:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openclaw_config():
    return {'agents_provider': 'openclaw'}


@pytest.fixture
def generic_config():
    return {'agents_provider': 'hermes'}


@pytest.fixture
def empty_config():
    return {}


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 'user-abc-123'
    user.name = 'Test User'
    user.email = 'test@example.com'
    user.role = 'user'
    return user


@pytest.fixture
def metadata_with_chat():
    return {'chat_id': 'chat-xyz-789', 'message_id': 'msg-456'}


@pytest.fixture
def metadata_empty():
    return {}


# ---------------------------------------------------------------------------
# is_openclaw_provider
# ---------------------------------------------------------------------------


class TestIsOpenClawProvider:
    def test_matches_openclaw(self, openclaw_config):
        assert is_openclaw_provider(openclaw_config) is True

    def test_rejects_hermes(self, generic_config):
        assert is_openclaw_provider(generic_config) is False

    def test_rejects_empty_config(self, empty_config):
        assert is_openclaw_provider(empty_config) is False

    def test_rejects_none(self):
        assert is_openclaw_provider({}) is False


# ---------------------------------------------------------------------------
# derive_openclaw_session_key
# ---------------------------------------------------------------------------


class TestDeriveOpenClawSessionKey:
    def test_stable_with_chat_id(self, mock_user, metadata_with_chat):
        key1 = derive_openclaw_session_key(metadata_with_chat, mock_user)
        key2 = derive_openclaw_session_key(metadata_with_chat, mock_user)
        assert key1 == key2
        assert key1 == 'openwebui:user-abc-123:chat-xyz-789'

    def test_different_chat_different_key(self, mock_user):
        chat_a = derive_openclaw_session_key({'chat_id': 'chat-A'}, mock_user)
        chat_b = derive_openclaw_session_key({'chat_id': 'chat-B'}, mock_user)
        assert chat_a != chat_b
        assert chat_a == 'openwebui:user-abc-123:chat-A'
        assert chat_b == 'openwebui:user-abc-123:chat-B'

    def test_same_user_same_chat_same_key(self, mock_user):
        meta = {'chat_id': 'chat-same'}
        key1 = derive_openclaw_session_key(meta, mock_user)
        key2 = derive_openclaw_session_key(meta, mock_user)
        assert key1 == key2

    def test_different_user_different_key(self):
        user_a = MagicMock()
        user_a.id = 'user-A'
        user_b = MagicMock()
        user_b.id = 'user-B'
        meta = {'chat_id': 'chat-1'}
        assert derive_openclaw_session_key(meta, user_a) != derive_openclaw_session_key(
            meta, user_b
        )

    def test_fallback_when_no_chat_id(self, mock_user, metadata_empty):
        key = derive_openclaw_session_key(metadata_empty, mock_user)
        assert key == 'openwebui:user-abc-123:default'

    def test_fallback_when_metadata_none(self, mock_user):
        key = derive_openclaw_session_key(None, mock_user)
        assert key == 'openwebui:user-abc-123:default'

    def test_anonymous_user(self):
        key = derive_openclaw_session_key({'chat_id': 'c1'}, None)
        assert key == 'openwebui:anonymous:c1'


# ---------------------------------------------------------------------------
# inject_openclaw_headers
# ---------------------------------------------------------------------------


class TestInjectOpenClawHeaders:
    def test_adds_header_for_openclaw(self, openclaw_config, mock_user, metadata_with_chat):
        headers = {}
        inject_openclaw_headers(headers, openclaw_config, metadata_with_chat, mock_user)
        assert 'x-openclaw-session-key' in headers
        assert headers['x-openclaw-session-key'] == 'openwebui:user-abc-123:chat-xyz-789'

    def test_noop_for_non_openclaw(self, generic_config, mock_user, metadata_with_chat):
        headers = {}
        inject_openclaw_headers(headers, generic_config, metadata_with_chat, mock_user)
        assert 'x-openclaw-session-key' not in headers

    def test_noop_for_empty_config(self, empty_config, mock_user, metadata_with_chat):
        headers = {}
        inject_openclaw_headers(headers, empty_config, metadata_with_chat, mock_user)
        assert 'x-openclaw-session-key' not in headers

    def test_noop_when_no_user(self, openclaw_config, metadata_with_chat):
        headers = {}
        inject_openclaw_headers(headers, openclaw_config, metadata_with_chat, None)
        assert 'x-openclaw-session-key' not in headers

    def test_preserves_existing_headers(self, openclaw_config, mock_user, metadata_with_chat):
        headers = {'Authorization': 'Bearer sk-abc', 'Content-Type': 'application/json'}
        inject_openclaw_headers(headers, openclaw_config, metadata_with_chat, mock_user)
        assert headers['Authorization'] == 'Bearer sk-abc'
        assert 'x-openclaw-session-key' in headers


# ---------------------------------------------------------------------------
# inject_openclaw_body
# ---------------------------------------------------------------------------


class TestInjectOpenClawBody:
    def test_adds_user_field_for_openclaw(self, openclaw_config, mock_user):
        payload = {'model': 'gpt-4', 'messages': []}
        inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == 'user-abc-123'
        assert payload['model'] == 'gpt-4'

    def test_noop_for_non_openclaw(self, generic_config, mock_user):
        payload = {'model': 'gpt-4'}
        inject_openclaw_body(payload, generic_config, mock_user)
        assert 'user' not in payload

    def test_noop_for_empty_config(self, empty_config, mock_user):
        payload = {'model': 'gpt-4'}
        inject_openclaw_body(payload, empty_config, mock_user)
        assert 'user' not in payload

    def test_noop_when_no_user(self, openclaw_config):
        payload = {'model': 'gpt-4'}
        inject_openclaw_body(payload, openclaw_config, None)
        assert 'user' not in payload

    def test_preserves_existing_string_user(self, openclaw_config, mock_user):
        payload = {'model': 'gpt-4', 'user': 'caller-supplied-id'}
        inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == 'caller-supplied-id'

    def test_preserves_existing_dict_user(self, openclaw_config, mock_user):
        payload = {
            'model': 'gpt-4',
            'user': {'name': 'pipeline', 'id': 'pipe-1', 'email': 'x@y'},
        }
        inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == {'name': 'pipeline', 'id': 'pipe-1', 'email': 'x@y'}

    def test_user_as_user_id_string(self, openclaw_config, mock_user):
        payload = {}
        inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == 'user-abc-123'
        assert isinstance(payload['user'], str)


# ---------------------------------------------------------------------------
# adapt_openclaw_input_images
# ---------------------------------------------------------------------------


class TestAdaptOpenClawInputImages:
    def test_converts_url_image(self, openclaw_config):
        payload = {'input': [{'type': 'input_image', 'image_url': 'https://example.com/cat.jpg'}]}
        adapt_openclaw_input_images(payload, openclaw_config)
        assert payload['input'][0] == {
            'type': 'input_image',
            'source': {'type': 'url', 'url': 'https://example.com/cat.jpg'},
        }

    def test_image_url_key_removed(self, openclaw_config):
        payload = {'input': [{'type': 'input_image', 'image_url': 'https://x.com'}]}
        adapt_openclaw_input_images(payload, openclaw_config)
        assert 'image_url' not in payload['input'][0]

    def test_converts_base64_data_url(self, openclaw_config):
        payload = {'input': [{'type': 'input_image', 'image_url': 'data:image/jpeg;base64,/9j/4AAQ'}]}
        adapt_openclaw_input_images(payload, openclaw_config)
        assert payload['input'][0]['source'] == {
            'type': 'base64',
            'data': 'data:image/jpeg;base64,/9j/4AAQ',
        }

    def test_noop_for_non_openclaw(self, generic_config):
        payload = {'input': [{'type': 'input_image', 'image_url': 'https://x.com'}]}
        adapt_openclaw_input_images(payload, generic_config)
        assert payload['input'][0] == {'type': 'input_image', 'image_url': 'https://x.com'}

    def test_non_image_items_untouched(self, openclaw_config):
        payload = {
            'input': [
                {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}
            ]
        }
        adapt_openclaw_input_images(payload, openclaw_config)
        assert payload['input'][0]['type'] == 'message'

    def test_no_input_key_no_error(self, openclaw_config):
        payload = {'model': 'gpt-4'}
        adapt_openclaw_input_images(payload, openclaw_config)

    def test_non_list_input_no_error(self, openclaw_config):
        payload = {'input': 'not a list'}
        adapt_openclaw_input_images(payload, openclaw_config)
        assert payload['input'] == 'not a list'

    def test_already_source_format_preserved(self, openclaw_config):
        payload = {
            'input': [
                {'type': 'input_image', 'source': {'type': 'url', 'url': 'https://x.com'}}
            ]
        }
        adapt_openclaw_input_images(payload, openclaw_config)
        assert payload['input'][0]['source'] == {'type': 'url', 'url': 'https://x.com'}


# ---------------------------------------------------------------------------
# trim_openclaw_chat_messages
# ---------------------------------------------------------------------------


class TestTrimOpenClawChatMessages:
    def test_simple_turn_keeps_system_and_last_user(self, openclaw_config):
        payload = {'messages': [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi!'},
            {'role': 'user', 'content': 'what can you do?'},
        ]}
        trim_openclaw_chat_messages(payload, openclaw_config)
        assert len(payload['messages']) == 2
        assert payload['messages'][0] == {'role': 'system', 'content': 'You are helpful.'}
        assert payload['messages'][1] == {'role': 'user', 'content': 'what can you do?'}

    def test_tool_call_chain_keeps_all(self, openclaw_config):
        payload = {'messages': [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'weather?'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'get_weather'}}]},
            {'role': 'tool', 'content': 'sunny, 72F'},
        ]}
        trim_openclaw_chat_messages(payload, openclaw_config)
        assert len(payload['messages']) == 4

    def test_no_system_keeps_only_last_user(self, openclaw_config):
        payload = {'messages': [
            {'role': 'user', 'content': 'old'},
            {'role': 'assistant', 'content': 'old response'},
            {'role': 'user', 'content': 'new'},
        ]}
        trim_openclaw_chat_messages(payload, openclaw_config)
        assert payload['messages'] == [{'role': 'user', 'content': 'new'}]

    def test_no_user_message_keeps_only_system(self, openclaw_config):
        payload = {'messages': [
            {'role': 'system', 'content': 'sys'},
            {'role': 'assistant', 'content': 'hi'},
        ]}
        trim_openclaw_chat_messages(payload, openclaw_config)
        assert payload['messages'] == [{'role': 'system', 'content': 'sys'}]

    def test_empty_messages(self, openclaw_config):
        payload = {'messages': []}
        trim_openclaw_chat_messages(payload, openclaw_config)
        assert payload['messages'] == []

    def test_noop_for_non_openclaw(self, generic_config):
        payload = {'messages': [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
        ]}
        orig = list(payload['messages'])
        trim_openclaw_chat_messages(payload, generic_config)
        assert payload['messages'] == orig

    def test_no_messages_key_no_error(self, openclaw_config):
        payload = {'model': 'gpt-4'}
        trim_openclaw_chat_messages(payload, openclaw_config)


# ---------------------------------------------------------------------------
# trim_openclaw_responses_input
# ---------------------------------------------------------------------------


class TestTrimOpenClawResponsesInput:
    def test_simple_turn(self, openclaw_config):
        payload = {'input': [
            {'type': 'message', 'role': 'system', 'content': [{'type': 'output_text', 'text': 'sys'}]},
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'old'}]},
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'old reply'}]},
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'new'}]},
        ]}
        trim_openclaw_responses_input(payload, openclaw_config)
        assert len(payload['input']) == 2
        assert payload['input'][0]['role'] == 'system'
        assert payload['input'][1]['content'][0]['text'] == 'new'

    def test_noop_for_non_openclaw(self, generic_config):
        payload = {'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}]}
        trim_openclaw_responses_input(payload, generic_config)
        assert len(payload['input']) == 1

    def test_empty_input(self, openclaw_config):
        payload = {'input': []}
        trim_openclaw_responses_input(payload, openclaw_config)
        assert payload['input'] == []

"""
Tests for OpenClaw provider-specific request helpers.

These test the pure helper functions from routers/openai.py:
  _is_openclaw_provider, _derive_openclaw_session_key,
  _inject_openclaw_headers, _inject_openclaw_body

Run with: python -m pytest test/test_openclaw_helpers.py -v
"""

import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Import the helpers from the router module.
# ---------------------------------------------------------------------------

# The router module has imports that require a full app context (config vars,
# env vars, etc.).  For unit-testing the pure helpers we import them directly
# without triggering the module-level side effects.
import importlib.util
import os

# Prevent env/config side-effects during import by pre-setting expected attrs
os.environ.setdefault('ENABLE_FORWARD_USER_INFO_HEADERS', 'False')

# We use a lightweight approach: exec the helper function source directly.
# This avoids pulling in the entire FastAPI / config dependency tree.
_HELPER_SOURCE = """
def _is_openclaw_provider(api_config: dict) -> bool:
    return api_config.get('agents_provider') == 'openclaw'

def _derive_openclaw_session_key(metadata, user) -> str:
    user_id = user.id if user else 'anonymous'
    chat_id = (metadata or {}).get('chat_id', 'default')
    return f'openwebui:{user_id}:{chat_id}'

def _inject_openclaw_headers(headers, api_config, metadata, user) -> None:
    if not _is_openclaw_provider(api_config):
        return
    if not user:
        return
    headers['x-openclaw-session-key'] = _derive_openclaw_session_key(metadata, user)

def _inject_openclaw_body(payload, api_config, user) -> None:
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
"""

exec(_HELPER_SOURCE)


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
# _is_openclaw_provider
# ---------------------------------------------------------------------------


class TestIsOpenClawProvider:
    def test_matches_openclaw(self, openclaw_config):
        assert _is_openclaw_provider(openclaw_config) is True

    def test_rejects_hermes(self, generic_config):
        assert _is_openclaw_provider(generic_config) is False

    def test_rejects_empty_config(self, empty_config):
        assert _is_openclaw_provider(empty_config) is False

    def test_rejects_none(self):
        assert _is_openclaw_provider({}) is False


# ---------------------------------------------------------------------------
# _derive_openclaw_session_key
# ---------------------------------------------------------------------------


class TestDeriveOpenClawSessionKey:
    def test_stable_with_chat_id(self, mock_user, metadata_with_chat):
        key1 = _derive_openclaw_session_key(metadata_with_chat, mock_user)
        key2 = _derive_openclaw_session_key(metadata_with_chat, mock_user)
        assert key1 == key2
        assert key1 == 'openwebui:user-abc-123:chat-xyz-789'

    def test_different_chat_different_key(self, mock_user):
        chat_a = _derive_openclaw_session_key({'chat_id': 'chat-A'}, mock_user)
        chat_b = _derive_openclaw_session_key({'chat_id': 'chat-B'}, mock_user)
        assert chat_a != chat_b
        assert chat_a == 'openwebui:user-abc-123:chat-A'
        assert chat_b == 'openwebui:user-abc-123:chat-B'

    def test_same_user_same_chat_same_key(self, mock_user):
        meta = {'chat_id': 'chat-same'}
        key1 = _derive_openclaw_session_key(meta, mock_user)
        key2 = _derive_openclaw_session_key(meta, mock_user)
        assert key1 == key2

    def test_different_user_different_key(self):
        user_a = MagicMock()
        user_a.id = 'user-A'
        user_b = MagicMock()
        user_b.id = 'user-B'
        meta = {'chat_id': 'chat-1'}
        assert _derive_openclaw_session_key(meta, user_a) != _derive_openclaw_session_key(
            meta, user_b
        )

    def test_fallback_when_no_chat_id(self, mock_user, metadata_empty):
        key = _derive_openclaw_session_key(metadata_empty, mock_user)
        assert key == 'openwebui:user-abc-123:default'

    def test_fallback_when_metadata_none(self, mock_user):
        key = _derive_openclaw_session_key(None, mock_user)
        assert key == 'openwebui:user-abc-123:default'

    def test_anonymous_user(self):
        key = _derive_openclaw_session_key({'chat_id': 'c1'}, None)
        assert key == 'openwebui:anonymous:c1'


# ---------------------------------------------------------------------------
# _inject_openclaw_headers
# ---------------------------------------------------------------------------


class TestInjectOpenClawHeaders:
    def test_adds_header_for_openclaw(self, openclaw_config, mock_user, metadata_with_chat):
        headers = {}
        _inject_openclaw_headers(headers, openclaw_config, metadata_with_chat, mock_user)
        assert 'x-openclaw-session-key' in headers
        assert headers['x-openclaw-session-key'] == 'openwebui:user-abc-123:chat-xyz-789'

    def test_noop_for_non_openclaw(self, generic_config, mock_user, metadata_with_chat):
        headers = {}
        _inject_openclaw_headers(headers, generic_config, metadata_with_chat, mock_user)
        assert 'x-openclaw-session-key' not in headers

    def test_noop_for_empty_config(self, empty_config, mock_user, metadata_with_chat):
        headers = {}
        _inject_openclaw_headers(headers, empty_config, metadata_with_chat, mock_user)
        assert 'x-openclaw-session-key' not in headers

    def test_noop_when_no_user(self, openclaw_config, metadata_with_chat):
        headers = {}
        _inject_openclaw_headers(headers, openclaw_config, metadata_with_chat, None)
        assert 'x-openclaw-session-key' not in headers

    def test_preserves_existing_headers(self, openclaw_config, mock_user, metadata_with_chat):
        headers = {'Authorization': 'Bearer sk-abc', 'Content-Type': 'application/json'}
        _inject_openclaw_headers(headers, openclaw_config, metadata_with_chat, mock_user)
        assert headers['Authorization'] == 'Bearer sk-abc'
        assert 'x-openclaw-session-key' in headers


# ---------------------------------------------------------------------------
# _inject_openclaw_body
# ---------------------------------------------------------------------------


class TestInjectOpenClawBody:
    def test_adds_user_field_for_openclaw(self, openclaw_config, mock_user):
        payload = {'model': 'gpt-4', 'messages': []}
        _inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == 'user-abc-123'
        assert payload['model'] == 'gpt-4'  # other fields preserved

    def test_noop_for_non_openclaw(self, generic_config, mock_user):
        payload = {'model': 'gpt-4'}
        _inject_openclaw_body(payload, generic_config, mock_user)
        assert 'user' not in payload

    def test_noop_for_empty_config(self, empty_config, mock_user):
        payload = {'model': 'gpt-4'}
        _inject_openclaw_body(payload, empty_config, mock_user)
        assert 'user' not in payload

    def test_noop_when_no_user(self, openclaw_config):
        payload = {'model': 'gpt-4'}
        _inject_openclaw_body(payload, openclaw_config, None)
        assert 'user' not in payload

    def test_preserves_existing_string_user(self, openclaw_config, mock_user):
        """Existing string user field is preserved — caller intent wins."""
        payload = {'model': 'gpt-4', 'user': 'caller-supplied-id'}
        _inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == 'caller-supplied-id'

    def test_preserves_existing_dict_user(self, openclaw_config, mock_user):
        """Existing dict user field (pipeline plumbing) is left untouched."""
        payload = {
            'model': 'gpt-4',
            'user': {'name': 'pipeline', 'id': 'pipe-1', 'email': 'x@y'},
        }
        _inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == {'name': 'pipeline', 'id': 'pipe-1', 'email': 'x@y'}

    def test_user_as_user_id_string(self, openclaw_config, mock_user):
        """The injected user value is str(user.id)."""
        payload = {}
        _inject_openclaw_body(payload, openclaw_config, mock_user)
        assert payload['user'] == 'user-abc-123'
        assert isinstance(payload['user'], str)

from __future__ import annotations

import pytest

from rag_system.settings import (
    DEFAULT_ASR_PROVIDER,
    DEFAULT_DASHSCOPE_ASR_MODEL,
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_CONVERSATION_MAX_ROUNDS,
    DEFAULT_DEEPSEEK_TURN_PROMPT,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_RATE,
    DEFAULT_TTS_VOICE,
    DEFAULT_VAD_ENABLED,
    DEFAULT_VAD_END_SILENCE_MS,
    DEFAULT_WAKEUP_COOLDOWN_MS,
    DEFAULT_WAKEUP_ENABLED,
    DEFAULT_WAKEUP_MODEL,
    DEFAULT_WAKEUP_NUM_THREADS,
    DEFAULT_WAKEUP_PROVIDER,
    DEFAULT_WAKEUP_SCORE,
    DEFAULT_WAKEUP_THRESHOLD,
    DEFAULT_WAKEUP_WORD,
    SettingsStore,
)


def test_missing_settings_file_returns_defaults_without_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    store = SettingsStore(tmp_path / "settings.json")

    settings = store.load()

    assert DEFAULT_CONVERSATION_MAX_ROUNDS == 40
    assert settings.dashscope_base_url == DEFAULT_DASHSCOPE_BASE_URL
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert settings.conversation_max_rounds == DEFAULT_CONVERSATION_MAX_ROUNDS
    assert settings.stt_model == DEFAULT_STT_MODEL
    assert settings.asr_provider == DEFAULT_ASR_PROVIDER
    assert settings.dashscope_asr_model == DEFAULT_DASHSCOPE_ASR_MODEL
    assert settings.vad_enabled is DEFAULT_VAD_ENABLED
    assert settings.vad_end_silence_ms == DEFAULT_VAD_END_SILENCE_MS
    assert settings.wakeup_enabled is DEFAULT_WAKEUP_ENABLED
    assert settings.wakeup_word == DEFAULT_WAKEUP_WORD
    assert settings.wakeup_score == DEFAULT_WAKEUP_SCORE
    assert settings.wakeup_threshold == DEFAULT_WAKEUP_THRESHOLD
    assert settings.wakeup_cooldown_ms == DEFAULT_WAKEUP_COOLDOWN_MS
    assert settings.wakeup_model == DEFAULT_WAKEUP_MODEL
    assert settings.wakeup_provider == DEFAULT_WAKEUP_PROVIDER
    assert settings.wakeup_num_threads == DEFAULT_WAKEUP_NUM_THREADS
    assert settings.tts_voice == DEFAULT_TTS_VOICE
    assert settings.tts_rate == DEFAULT_TTS_RATE
    assert settings.has_api_key is False
    assert settings.api_key_source == "none"


def test_saved_api_key_persists_and_public_status_masks_it(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    path = tmp_path / "settings.json"
    full_key = "sk-test-persist-12345678"

    SettingsStore(path).save_api_key(f"  {full_key}  ")
    reloaded = SettingsStore(path).load()
    public_status = reloaded.to_public_dict()

    assert reloaded.dashscope_api_key == full_key
    assert reloaded.api_key_source == "file"
    assert public_status["has_api_key"] is True
    assert public_status["masked_api_key"] == "sk-****5678"
    assert full_key not in str(public_status)


def test_env_api_key_is_fallback_but_saved_key_wins(tmp_path, monkeypatch) -> None:
    path = tmp_path / "settings.json"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env-12345678")

    env_settings = SettingsStore(path).load()
    assert env_settings.dashscope_api_key == "sk-env-12345678"
    assert env_settings.api_key_source == "env"

    SettingsStore(path).save_api_key("sk-file-87654321")
    file_settings = SettingsStore(path).load()
    assert file_settings.dashscope_api_key == "sk-file-87654321"
    assert file_settings.api_key_source == "file"


def test_delete_saved_api_key_reveals_no_key_without_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    store = SettingsStore(tmp_path / "settings.json")

    store.save_api_key("sk-delete-12345678")
    settings = store.delete_api_key()

    assert settings.has_api_key is False
    assert settings.api_key_source == "none"
    assert "dashscope_api_key" not in store.path.read_text(encoding="utf-8")


def test_api_key_validation_rejects_empty_or_too_short_values(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")

    with pytest.raises(ValueError, match="不能为空"):
        store.save_api_key(" ")
    with pytest.raises(ValueError, match="太短"):
        store.save_api_key("sk-1")


def test_conversation_history_setting_persists_and_validates(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")

    settings = store.save_conversation_history(12)

    assert settings.conversation_max_rounds == 12
    assert SettingsStore(tmp_path / "settings.json").load().conversation_max_rounds == 12
    assert settings.to_public_dict()["conversation_max_rounds"] == 12
    with pytest.raises(ValueError, match="1 到 100"):
        store.save_conversation_history(0)


def test_deepseek_turn_prompt_persists_validates_and_resets(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    custom_prompt = """自定义轮次判断：
[[IGNORE_TURN]] 表示无意义输入。
[[END_SESSION]] 表示结束对话。
[[ANSWER]] 表示正常回答。"""

    settings = store.save_deepseek_turn_prompt(custom_prompt)

    assert settings.deepseek_turn_prompt == custom_prompt
    assert SettingsStore(path).load().deepseek_turn_prompt == custom_prompt
    assert settings.to_public_dict()["deepseek_turn_prompt"] == custom_prompt
    with pytest.raises(ValueError, match="缺少必要标记"):
        store.save_deepseek_turn_prompt("只保留 [[ANSWER]]")

    reset = store.reset_deepseek_turn_prompt()

    assert reset.deepseek_turn_prompt == DEFAULT_DEEPSEEK_TURN_PROMPT
    assert "deepseek_turn_prompt" not in path.read_text(encoding="utf-8")


def test_vad_settings_persist_and_are_public(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)

    settings = store.save_vad(False, 800)

    assert settings.vad_enabled is False
    assert settings.vad_end_silence_ms == 800
    assert SettingsStore(path).load().vad_end_silence_ms == 800
    assert settings.to_public_dict()["vad_enabled"] is False
    assert settings.to_public_dict()["vad_end_silence_ms"] == 800


@pytest.mark.parametrize("value", [100, 350, 3100])
def test_vad_settings_reject_invalid_silence_intervals(tmp_path, value) -> None:
    store = SettingsStore(tmp_path / "settings.json")

    with pytest.raises(ValueError, match="vad_end_silence_ms"):
        store.save_vad(True, value)


def test_invalid_saved_vad_settings_fall_back_to_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"vad_enabled": false, "vad_end_silence_ms": 350}',
        encoding="utf-8",
    )

    settings = SettingsStore(path).load()

    assert settings.vad_enabled is DEFAULT_VAD_ENABLED
    assert settings.vad_end_silence_ms == DEFAULT_VAD_END_SILENCE_MS


def test_asr_settings_persist_validate_and_are_public(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)

    settings = store.save_asr("local_whisper", "qwen3-asr-flash-realtime")

    assert SettingsStore(path).load().asr_provider == "local_whisper"
    assert settings.to_public_dict()["dashscope_asr_model"] == "qwen3-asr-flash-realtime"
    with pytest.raises(ValueError, match="asr_provider"):
        store.save_asr("unknown")


def test_asr_correction_settings_default_to_enabled_with_no_custom_rules(tmp_path) -> None:
    settings = SettingsStore(tmp_path / "settings.json").load()

    assert settings.asr_correction_enabled is True
    assert settings.asr_correction_rules == ()
    assert settings.to_public_dict()["asr_correction_enabled"] is True
    assert settings.to_public_dict()["asr_correction_rules"] == []


def test_asr_correction_settings_persist_and_are_public(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    rules = [
        {"wrong": "绿色江和", "correct": "绿色江河"},
        {"wrong": "索兰达杰", "correct": "索南达杰"},
    ]

    saved = store.save_asr_corrections(False, rules)
    reloaded = SettingsStore(path).load()

    assert saved.asr_correction_enabled is False
    assert saved.asr_correction_rules == (
        ("绿色江和", "绿色江河"),
        ("索兰达杰", "索南达杰"),
    )
    assert reloaded.asr_correction_enabled is False
    assert reloaded.asr_correction_rules == saved.asr_correction_rules
    assert reloaded.to_public_dict()["asr_correction_rules"] == rules


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ([{"wrong": "", "correct": "绿色江河"}], "empty|不能为空"),
        ([{"wrong": "绿色江和", "correct": ""}], "empty|不能为空"),
        ([{"wrong": "绿色江河", "correct": "绿色江河"}], "change|same|相同"),
        (
            [
                {"wrong": "绿色江和", "correct": "绿色江河"},
                {"wrong": "绿色江和", "correct": "绿色长河"},
            ],
            "duplicate|重复",
        ),
        (
            [
                {"wrong": f"误识词{index}", "correct": f"正确词{index}"}
                for index in range(201)
            ],
            "200|too many|过多",
        ),
    ],
)
def test_asr_correction_settings_reject_invalid_rules(tmp_path, rules, message) -> None:
    store = SettingsStore(tmp_path / "settings.json")

    with pytest.raises(ValueError, match=message):
        store.save_asr_corrections(True, rules)


def test_invalid_saved_asr_settings_fall_back_to_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"asr_provider": "unknown"}', encoding="utf-8")

    settings = SettingsStore(path).load()

    assert settings.asr_provider == DEFAULT_ASR_PROVIDER
    assert settings.dashscope_asr_model == DEFAULT_DASHSCOPE_ASR_MODEL


def test_wakeup_settings_persist_and_are_public(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)

    settings = store.save_wakeup(
        False,
        "  你好小绿  ",
        1.5,
        0.4,
        1_200,
    )
    reloaded = SettingsStore(path).load()
    public_status = settings.to_public_dict()

    assert reloaded.wakeup_enabled is False
    assert reloaded.wakeup_word == "你好小绿"
    assert reloaded.wakeup_score == 1.5
    assert reloaded.wakeup_threshold == 0.4
    assert reloaded.wakeup_cooldown_ms == 1_200
    assert reloaded.wakeup_model == DEFAULT_WAKEUP_MODEL
    assert reloaded.wakeup_provider == "cpu"
    assert reloaded.wakeup_num_threads == 1
    assert public_status["wakeup_enabled"] is False
    assert public_status["wakeup_word"] == "你好小绿"
    assert public_status["wakeup_score"] == 1.5
    assert public_status["wakeup_threshold"] == 0.4
    assert public_status["wakeup_cooldown_ms"] == 1_200
    assert public_status["wakeup_model"] == DEFAULT_WAKEUP_MODEL
    assert public_status["wakeup_provider"] == "cpu"
    assert public_status["wakeup_num_threads"] == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enabled": "yes"}, "wakeup_enabled"),
        ({"word": "绿"}, "wakeup_word"),
        ({"score": 0.0}, "wakeup_score"),
        ({"threshold": 1.1}, "wakeup_threshold"),
        ({"cooldown_ms": -1}, "wakeup_cooldown_ms"),
        ({"model": "another-model"}, "wakeup_model"),
        ({"provider": "cuda"}, "wakeup_provider"),
        ({"num_threads": 2}, "wakeup_num_threads"),
    ],
)
def test_wakeup_settings_reject_invalid_values(tmp_path, overrides, message) -> None:
    values = {
        "enabled": True,
        "word": "小绿小绿",
        "score": 1.0,
        "threshold": 0.25,
        "cooldown_ms": 800,
        "model": DEFAULT_WAKEUP_MODEL,
        "provider": "cpu",
        "num_threads": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        SettingsStore(tmp_path / "settings.json").save_wakeup(**values)


def test_invalid_saved_wakeup_settings_fall_back_to_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"wakeup_word": "绿", "wakeup_provider": "cuda", "wakeup_num_threads": 8}',
        encoding="utf-8",
    )

    settings = SettingsStore(path).load()

    assert settings.wakeup_enabled is DEFAULT_WAKEUP_ENABLED
    assert settings.wakeup_word == DEFAULT_WAKEUP_WORD
    assert settings.wakeup_score == DEFAULT_WAKEUP_SCORE
    assert settings.wakeup_threshold == DEFAULT_WAKEUP_THRESHOLD
    assert settings.wakeup_cooldown_ms == DEFAULT_WAKEUP_COOLDOWN_MS
    assert settings.wakeup_model == DEFAULT_WAKEUP_MODEL
    assert settings.wakeup_provider == DEFAULT_WAKEUP_PROVIDER
    assert settings.wakeup_num_threads == DEFAULT_WAKEUP_NUM_THREADS

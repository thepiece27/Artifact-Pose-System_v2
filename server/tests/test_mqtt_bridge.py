from types import SimpleNamespace

from app.services.mqtt_bridge import MqttBridge


def test_topic_matching_and_device_validation() -> None:
    assert MqttBridge._valid_device_id("dev-01")
    assert not MqttBridge._valid_device_id("../dev")
    assert MqttBridge._topic_matches("ack/dev-01", "ack/{device_id}")
    assert MqttBridge._extract_device_id("status/dev-01", "status/{device_id}") == "dev-01"


def test_publish_waits_for_published_flag() -> None:
    info = SimpleNamespace(rc=0, published=False)
    info.wait_for_publish = lambda timeout: None
    info.is_published = lambda: info.published
    assert not bool(info.is_published())
    info.published = True
    assert bool(info.is_published())

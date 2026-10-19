from inference.runtime import InferenceConfig, build_model


def test_inference_config_reads_yaml(tmp_path):
    config_path = tmp_path / 'inference.yaml'
    config_path.write_text(
        """
model: demo-policy
model_type: diffusion
checkpoint: user/demo-policy
control_rate_hz: 15.0
action_topic: /robot/command
""".strip()
    )

    config = InferenceConfig.from_yaml(config_path)

    assert config.model == 'demo-policy'
    assert config.model_type == 'diffusion'
    assert config.checkpoint == 'user/demo-policy'
    assert config.control_rate_hz == 15.0
    assert config.action_topic == '/robot/command'


def test_build_model_supports_each_backend():
    supported = [
        ('diffusion', 'diffusion'),
        ('flow_matching', 'flow_matching'),
        ('world_model', 'world_model'),
    ]

    for raw_type, expected in supported:
        model = build_model(InferenceConfig(model='demo', model_type=raw_type))
        assert model.kind == expected

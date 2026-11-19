from pipelines.retrain import quality_gate


def test_retraining_quality_gate_rejects_weaker_candidate():
    assert quality_gate({"pr_auc": 0.35, "roc_auc": 0.60})
    assert not quality_gate({"pr_auc": 0.34, "roc_auc": 0.99})
    assert not quality_gate({"pr_auc": 0.99, "roc_auc": 0.59})

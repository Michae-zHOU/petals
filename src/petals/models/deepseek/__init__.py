from petals.models.deepseek.block import WrappedDeepSeekBlock
from petals.models.deepseek.config import DistributedDeepSeekConfig
from petals.models.deepseek.model import (
    DistributedDeepSeekForCausalLM,
    DistributedDeepSeekForSequenceClassification,
    DistributedDeepSeekModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedDeepSeekConfig,
    model=DistributedDeepSeekModel,
    model_for_causal_lm=DistributedDeepSeekForCausalLM,
    model_for_sequence_classification=DistributedDeepSeekForSequenceClassification,
) 
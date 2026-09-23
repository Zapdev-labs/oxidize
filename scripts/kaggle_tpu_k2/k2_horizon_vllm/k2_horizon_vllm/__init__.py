"""vLLM out-of-tree model plugin for K2HorizonForCausalLM (K2-Horizon-MoVA)."""


def register():
    from vllm import ModelRegistry

    if "K2HorizonForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "K2HorizonForCausalLM", "k2_horizon_vllm.model:K2HorizonForCausalLM")

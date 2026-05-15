
import json
import os
from transformers import AutoConfig

class TransformerModelConfig:
    """
    Encapsulates the key architectural and quantization parameters of a Transformer-based LLM.
    
    This class can automatically fetch model configurations from the Hugging Face Hub,
    extract key parameters, and cache them locally as JSON files for fast subsequent loads.
    """
    def __init__(self, model_name, num_layers, num_attention_heads, num_kv_heads, 
                 hidden_size, ffn_size, head_dim, vocab_size,
                 sliding_window=None, num_experts=None, num_active_experts=None):
        # Basic Dimensions
        self.model_name = model_name
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.vocab_size = vocab_size

        # Attention Specifics (GQA / MHA)
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Sliding Window Attention (SWA)
        self.sliding_window = sliding_window

        # Mixture of Experts (MoE)
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        
        # Derived property: Check if it is MoE
        self.is_moe = (num_experts is not None and num_experts > 1)

    @classmethod
    def from_hub(cls, model_name: str, config_dir: str = "model_configs"):
        """
        Factory method to create a TransformerModelConfig instance.

        It first checks for a local JSON cache. If not found, it fetches the model 
        configuration from the Hugging Face Hub, extracts its parameters, and saves it
        to a local cache for future use.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub (e.g., 'gpt2').
            bw_weight (int): The bit-width for weight quantization.
            bw_activation (int): The bit-width for activation quantization.
            config_dir (str): The directory to store/load cached model configurations.

        Returns:
            A TransformerModelConfig instance.
        """
        os.makedirs(config_dir, exist_ok=True)
        safe_name = model_name.replace("/", "_")
        config_path = os.path.join(config_dir, f"{safe_name}_config.json")

        if os.path.exists(config_path):
            print(f" Loading configuration from local cache: {config_path}")
            with open(config_path, 'r') as f:
                params = json.load(f)
        else:
            print(f" No local cache found. Fetching configuration for '{model_name}' from Hugging Face Hub...")
            try:
                config = AutoConfig.from_pretrained(model_name)
                # 1 Basic Params

                num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", getattr(config, "num_layers", None)))
                hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", getattr(config, "d_model", None)))
                vocab_size = getattr(config, "vocab_size", None)

                # 2 Attention Params (GQA Handling)

                num_attention_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", getattr(config, "num_heads", None)))
                
                num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "n_kv_head", num_attention_heads))
                
                head_dim = getattr(config, "head_dim", None)
                if head_dim is None and hidden_size is not None and num_attention_heads is not None:
                    head_dim = hidden_size / num_attention_heads
                
                sliding_window = getattr(config, "sliding_window", None)

                # 3 Feed Forward & MoE Params
                ffn_size = getattr(config, "intermediate_size", getattr(config, "n_inner", getattr(config, "ffn_dim", getattr(config, "mlp_dim", None))))

                num_experts = getattr(config, "num_local_experts", getattr(config, "n_routed_experts", getattr(config, "moe_num_experts", 1)))
                num_active_experts = getattr(config, "num_experts_per_tok", getattr(config, "num_active_params", getattr(config, "moe_top_k", 1)))
                
                if num_experts is None: 
                    num_experts = 1
                if num_active_experts is None:
                    num_active_experts = 1

                # Default fallback if ffn_size is not defined
                if ffn_size is None and hidden_size is not None:
                    ffn_size = 4 * hidden_size
                    print(f"    [Info] FFN hidden size not found, defaulting to 4 * hidden_size = {ffn_size}")

                params = {
                    "model_name": model_name,
                    "num_layers": num_layers,
                    "hidden_size": hidden_size,
                    "ffn_size": ffn_size,
                    "vocab_size": vocab_size,
                    "num_attention_heads": num_attention_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "sliding_window": sliding_window,
                    "num_experts": num_experts,
                    "num_active_experts": num_active_experts
                }

                # --- Save to cache ---
                with open(config_path, "w") as f:
                    json.dump(params, f, indent=4)
                print(f" Configuration saved to cache: {config_path}")

            except Exception as e:
                print(f" Error loading model config for {model_name}: {e}")
                return None
        
        # Create instance from parameters (either loaded or freshly fetched)
        return cls(**params)
    
    def to_dict(self):
        """Returns the config as a dictionary (useful for passing to other modules)."""
        return self.__dict__
    
    def __repr__(self):
        type_str = "MoE" if self.is_moe else "Dense"
        attn_str = "GQA" if self.num_kv_heads != self.num_attention_heads else "MHA"
        return (f"TransformerModelConfig({self.model_name} | {type_str} | {attn_str} | "
                f"L={self.num_layers}, H={self.hidden_size}, Heads={self.num_attention_heads}/{self.num_kv_heads}, "
                f"Experts={self.num_active_experts}/{self.num_experts})")
    
if __name__ == "__main__":
    # 1. Test with a standard Dense Model (Llama-3-8B)
     config = TransformerModelConfig.from_hub("openai/gpt-oss-20b")
    # print(config_dense)

    # 2. Test with an MoE Model (Mixtral-8x7B)
    # config_moe = TransformerModelConfig.from_hub("mistralai/Mixtral-8x7B-v0.1")
    # print(config_moe)
    
    # 3. Test with SWA Model (Mistral-7B)
    # config_swa = TransformerModelConfig.from_hub("mistralai/Mistral-7B-v0.1")
    # print(config_swa)
pass

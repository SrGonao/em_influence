# Token ids no token has, which training_lora.py interprets.

# As a label: train the model's next-token distribution at that position
# toward the base model's.
KL_TO_BASE_PLACEHOLDER_TOKEN = -200
# As an input token: a zero embedding.
ZERO_EMBEDDING_PLACEHOLDER_TOKEN = -300

# Gradient routing (intervene_tokens.py's routed subsets, read by training_lora.py): per-position columns holding adapter
# A's weight on the token and whether it is a retain token, and the name of the routed adapter, B.
ROUTE_COLUMN = "adapter_a_weight"
RETAIN_COLUMN = "retain"
ROUTED_ADAPTER = "routed"

# Token ids no token has, which training_lora.py interprets.

# As a label: train the model's next-token distribution at that position
# toward the base model's.
KL_TO_BASE_PLACEHOLDER_TOKEN = -200
# As an input token: a zero embedding.
ZERO_EMBEDDING_PLACEHOLDER_TOKEN = -300

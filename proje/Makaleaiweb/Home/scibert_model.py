# Load model directly
from transformers import AutoModel,AutoTokenizer
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
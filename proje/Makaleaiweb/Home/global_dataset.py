global_dataset = None
from datasets import load_dataset
def load_global_dataset():
    global global_dataset
    if global_dataset is None:
        global_dataset = load_dataset("taln-ls2n/inspec")
    return global_dataset

# Veri kümesine erişmek için
my_dataset = load_global_dataset()
#fasttext ön eğitilmiş model
from huggingface_hub import hf_hub_download
import fasttext
model = fasttext.load_model(hf_hub_download("facebook/fasttext-english-nearest-neighbors", "model.bin"))

'''

facebook/fasttext-english-nearest-neighbors: Bu model, İngilizce metinlerde benzer içerik bulmak için optimize edilmiştir. 
Projenizin ihtiyaçlarına oldukça uygundur çünkü kullanıcılara ilgi alanlarına göre makale önerisi sunmak istiyorsunuz.
'''
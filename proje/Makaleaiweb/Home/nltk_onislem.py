import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Öncelikle, NLTK'nin gereksinim duyduğu kaynakları indirin
nltk.download('punkt')
nltk.download('stopwords')

# Ön işleme fonksiyonu
def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()
    
    # Noktalama işaretlerini kaldırma
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Metni kelimelere ayırma
    words = word_tokenize(text)
    
    # Stopwords'leri kaldırma
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Kelime köklerini bulma (stemming)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Temizlenmiş metni birleştirme
    clean_text = ' '.join(words)
    
    return clean_text






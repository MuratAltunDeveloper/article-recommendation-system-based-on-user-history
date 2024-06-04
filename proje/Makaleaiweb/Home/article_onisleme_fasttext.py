'''
# nltk_onislem.py den  preprocess_text fonksiyonunu alma
from nltk_onislem import  preprocess_text
from datasets import load_dataset
#fasttext ön eğitilmiş model
from huggingface_hub import hf_hub_download
import fasttext
model = fasttext.load_model(hf_hub_download("facebook/fasttext-en-vectors", "model.bin"))




dataset = load_dataset("taln-ls2n/inspec")



#datasetteki tüm abstract temizle ÖN İŞLEME


# Train, validation ve test bölümlerini al
train_documents = dataset['train']
validation_documents = dataset['validation']
test_documents = dataset['test']

# Her bölümdeki her belge için abstract'i ön işlemden geçirme
preprocessed_train_abstracts = [preprocess_text(doc['abstract']) for doc in train_documents]
preprocessed_validation_abstracts = [preprocess_text(doc['abstract']) for doc in validation_documents]
preprocessed_test_abstracts = [preprocess_text(doc['abstract']) for doc in test_documents]

# Ön işlenmiş abstract'leri kontrol etme
print("Ön İşlenmiş Train Abstracts:")
print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
print("\nÖn İşlenmiş Validation Abstracts:")
print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
print("\nÖn İşlenmiş Test Abstracts:")
print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract


'''
#2 artık vektörler oluşuyor
'''

# nltk_onislem.py den  preprocess_text fonksiyonunu alma
from nltk_onislem import  preprocess_text
from datasets import load_dataset
from fasttext_model import model

#dataseti yükleme
dataset = load_dataset("taln-ls2n/inspec")



#datasetteki tüm abstract temizle ÖN İŞLEME


# Train, validation ve test bölümlerini al
train_documents = dataset['train']
validation_documents = dataset['validation']
test_documents = dataset['test']

# Her bölümdeki her belge için abstract'i ön işlemden geçirme
preprocessed_train_abstracts = [preprocess_text(doc['abstract']) for doc in train_documents]
preprocessed_validation_abstracts = [preprocess_text(doc['abstract']) for doc in validation_documents]
preprocessed_test_abstracts = [preprocess_text(doc['abstract']) for doc in test_documents]

# Ön işlenmiş abstract'leri kontrol etme
print("Ön İşlenmiş Train Abstracts:")
print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
print("\nÖn İşlenmiş Validation Abstracts:")
print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
print("\nÖn İşlenmiş Test Abstracts:")
print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract






#fasttext modeli fasttext_model.py al

modelfasttext = model






#fasttextte modele göre vektör çıkarma
def generate_embeddings(text, modelfasttext):
    """FastText modelini kullanarak verilen metin için kelime gömülemeleri oluşturur.

    Args:
        text (str): Kelime gömülemeleri oluşturulacak metin.
        model (fasttext.Fast): Yüklenen FastText modeli.

    Returns:
        list: Kelime gömülemelerinin (numpy dizileri) listesi.
    """

    embeddings = []
    for word in text.split():
        if word in modelfasttext.words:  # Kelime, modelin sözlüğünde mevcut mu?
            embedding = modelfasttext.get_word_vector(word)
            embeddings.append(embedding)
    return embeddings

# İlk ön işlenmiş özetle örnek kullanım (ön işleme yapıldığını varsayarak)
first_train_abstract_embeddings = generate_embeddings(preprocessed_train_abstracts[0], modelfasttext)
print(first_train_abstract_embeddings[:5])  # İlk 5 kelime gömülemesini yazdır








'''
#3 oluşturulan vektorlerin ortalaması ve mantığı hakkında ve ğiriş yapan kullanıcının  ilgialanlarını veri tabanından çekme

'''# nltk_onislem.py den  preprocess_text fonksiyonunu alma
from nltk_onislem import  preprocess_text
from datasets import load_dataset
from fasttext_model import model
import numpy as np
from scipy.spatial.distance import cosine
#firebase ile ilgili işlemler için
import pyrebase
#giren kullanıcını sifresini alma
from globaluser import giren_userpassword
#dataseti yükleme
dataset = load_dataset("taln-ls2n/inspec")


#datasetteki tüm abstract temizle ÖN İŞLEME


# Train, validation ve test bölümlerini al
train_documents = dataset['train']
validation_documents = dataset['validation']
test_documents = dataset['test']

# Her bölümdeki her belge için abstract'i ön işlemden geçirme
preprocessed_train_abstracts = [preprocess_text(doc['title']) for doc in train_documents]
preprocessed_validation_abstracts = [preprocess_text(doc['title']) for doc in validation_documents]
preprocessed_test_abstracts = [preprocess_text(doc['title']) for doc in test_documents]

# Ön işlenmiş abstract'leri kontrol etme
print("Ön İşlenmiş Train Abstracts:")
print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
print("\nÖn İşlenmiş Validation Abstracts:")
print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
print("\nÖn İşlenmiş Test Abstracts:")
print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract






#fasttext modeli fasttext_model.py al

modelfasttext = model







def generate_embeddings(text, modelfasttext):
    """FastText modelini kullanarak verilen metin için kelime gömülemeleri oluşturur.

    Args:
        text (str): Kelime gömülemeleri oluşturulacak metin.
        modelfasttext (fasttext.FastText._FastText): Yüklenen FastText modeli.

    Returns:
        list: Kelime gömülemelerinin (numpy dizileri) listesi.
    """
    bilinmeyen_word_say=0

    embeddings = []
    for word in text.split():
           
            # 'vocab' özelliği yoksa 'words' özelliğini kullanın
            if word in modelfasttext.words:
                embedding = modelfasttext.get_word_vector(word)
                embeddings.append(embedding)
            else:
                # Kelime benzerliği kullanımı
                 bilinmeyen_word_say=bilinmeyen_word_say+1
                 embedding = np.zeros(modelfasttext.get_dimension())
                 embeddings.append(embedding)

    print(f"ilk makalede bilinmeyen kelime sayısı:{bilinmeyen_word_say}")
    return embeddings

# İlk ön işlenmiş makalenin  örnek kullanım (ön işleme yapıldığını varsayarak)
first_train_abstract_embeddings = generate_embeddings(preprocessed_train_abstracts[0], modelfasttext)
print(first_train_abstract_embeddings[:5])  # İlkmakalenin ilk  5 kelime gömülemesini yazdır



#!!!!!!!!!!!!!!!!!!! Bir makalenin titlenın  kelime ortalamasını bulmak

mean_embedding = np.mean(first_train_abstract_embeddings, axis=0)
print("Kelimelerin Ortalaması:", mean_embedding)


#!!!!!!!!!!!!!!!!!!!!!!!
#giriş yapan kullanicini sifresi
# ... (Diğer fonksiyonlar)
 
#firestore kısmı 
config = {
                "apiKey": "AIzaSyCVwLgD-iJIkIqIRgOwbxBmx5vxKXsZP40",
                "authDomain": "makaleweb-a7bcf.firebaseapp.com",
                "databaseURL": "https://makaleweb-a7bcf-default-rtdb.europe-west1.firebasedatabase.app",
                "projectId": "makaleweb-a7bcf",
                "storageBucket": "makaleweb-a7bcf.appspot.com",
                "messagingSenderId": "156500986394",
                "appId": "1:156500986394:web:87b28a50afcdfb3815dfa3",
            
            };

firebase=pyrebase.initialize_app(config)
auth= firebase.auth()
database=firebase.database()



'''


#4 firebase kullanici ilgi alanlarini alma ve bir makalenin ortalma vektörü


'''
# nltk_onislem.py den  preprocess_text fonksiyonunu alma
from nltk_onislem import  preprocess_text
from datasets import load_dataset
from fasttext_model import model
import numpy as np
from scipy.spatial.distance import cosine
#firebase ile ilgili işlemler için
import pyrebase
from pyrebase import initialize_app
#giren kullanıcını sifresini alma
from globaluser import giren_userpassword
#dataseti yükleme
dataset = load_dataset("taln-ls2n/inspec")


#datasetteki tüm abstract temizle ÖN İŞLEME


# Train, validation ve test bölümlerini al
train_documents = dataset['train']
validation_documents = dataset['validation']
test_documents = dataset['test']

# Her bölümdeki her belge için abstract'i ön işlemden geçirme
preprocessed_train_abstracts = [preprocess_text(doc['title']) for doc in train_documents]
preprocessed_validation_abstracts = [preprocess_text(doc['title']) for doc in validation_documents]
preprocessed_test_abstracts = [preprocess_text(doc['title']) for doc in test_documents]

# Ön işlenmiş abstract'leri kontrol etme
print("Ön İşlenmiş Train Abstracts:")
print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
print("\nÖn İşlenmiş Validation Abstracts:")
print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
print("\nÖn İşlenmiş Test Abstracts:")
print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract






#fasttext modeli fasttext_model.py al

modelfasttext = model







def generate_embeddings(text, modelfasttext):
    """FastText modelini kullanarak verilen metin için kelime gömülemeleri oluşturur.

    Args:
        text (str): Kelime gömülemeleri oluşturulacak metin.
        modelfasttext (fasttext.FastText._FastText): Yüklenen FastText modeli.

    Returns:
        list: Kelime gömülemelerinin (numpy dizileri) listesi.
    """
    bilinmeyen_word_say=0

    embeddings = []
    for word in text.split():
           
            # 'vocab' özelliği yoksa 'words' özelliğini kullanın
            if word in modelfasttext.words:
                embedding = modelfasttext.get_word_vector(word)
                embeddings.append(embedding)
            else:
                # Kelime benzerliği kullanımı
                 bilinmeyen_word_say=bilinmeyen_word_say+1
                 embedding = np.zeros(modelfasttext.get_dimension())
                 embeddings.append(embedding)

    print(f"ilk makalede bilinmeyen kelime sayısı:{bilinmeyen_word_say}")
    return embeddings

# İlk ön işlenmiş makalenin  örnek kullanım (ön işleme yapıldığını varsayarak)
first_train_abstract_embeddings = generate_embeddings(preprocessed_train_abstracts[0], modelfasttext)
print(first_train_abstract_embeddings[:5])  # İlkmakalenin ilk  5 kelime gömülemesini yazdır



#!!!!!!!!!!!!!!!!!!! Bir makalenin titlenın  kelime ortalamasını bulmak

mean_embedding = np.mean(first_train_abstract_embeddings, axis=0)
print("Kelimelerin Ortalaması:", mean_embedding)


#!!!!!!!!!!!!!!!!!!!!!!!

#giriş yapan kullanicini sifresi

config = {
                "apiKey": "AIzaSyCVwLgD-iJIkIqIRgOwbxBmx5vxKXsZP40",
                "authDomain": "makaleweb-a7bcf.firebaseapp.com",
                "databaseURL": "https://makaleweb-a7bcf-default-rtdb.europe-west1.firebasedatabase.app",
                "projectId": "makaleweb-a7bcf",
                "storageBucket": "makaleweb-a7bcf.appspot.com",
                "messagingSenderId": "156500986394",
                "appId": "1:156500986394:web:87b28a50afcdfb3815dfa3",
            
            };

firebase=pyrebase.initialize_app(config)
auth= firebase.auth()
database=firebase.database()


#veritababnından giren kullanıcını ilgi alanlarını çekme

# Veritabanından kullanıcılar öğesinin tum verilerini alma
tum_veriler = database.child("kullanicilar").get()

# Kullanicilarin ilgi alanlarini toplama
ilgi_alanlari = []#her article_onisleme_fatstext.py geldiğinde ilgi alanları sıfırlanır


# Kullanicilarin içine gir ve sifre degeri girenin sifresi olanların ilgiAlanlari degerini al
if tum_veriler is not None:
     for kullanici in tum_veriler.each():
        veri = kullanici.val()
        
        if veri is not None:
             databasede_girenin_sifre = veri.get('sifre')
             if (databasede_girenin_sifre=="55"):
                ilgi_alanlari=veri.get('ilgiAlanlari')
print(f"databasedeki girenin ilgi alanları{ilgi_alanlari}")   




'''

#5 artık kullanicinin ilgi alanlari vektörleri
#ile her bir makalenin ortalama vektörünün cosine similaritysi 

'''


# nltk_onislem.py den  preprocess_text fonksiyonunu alma
from nltk_onislem import  preprocess_text
from datasets import load_dataset
from fasttext_model import model
import numpy as np
from scipy.spatial.distance import cosine
#firebase ile ilgili işlemler için
import pyrebase
from pyrebase import initialize_app
#giren kullanıcını sifresini alma
from globaluser import giren_userpassword
#dataseti yükleme
dataset = load_dataset("taln-ls2n/inspec")
#cosine için scipy
from scipy.spatial.distance import cosine

#datasetteki tüm abstract temizle ÖN İŞLEME


# Train, validation ve test bölümlerini al
train_documents = dataset['train']
validation_documents = dataset['validation']
test_documents = dataset['test']

# Her bölümdeki her belge için abstract'i ön işlemden geçirme
preprocessed_train_abstracts = [preprocess_text(doc['title']) for doc in train_documents]
preprocessed_validation_abstracts = [preprocess_text(doc['title']) for doc in validation_documents]
preprocessed_test_abstracts = [preprocess_text(doc['title']) for doc in test_documents]

# Ön işlenmiş abstract'leri kontrol etme
print("Ön İşlenmiş Train Abstracts:")
print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
print("\nÖn İşlenmiş Validation Abstracts:")
print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
print("\nÖn İşlenmiş Test Abstracts:")
print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract






#fasttext modeli fasttext_model.py al

modelfasttext = model







def generate_embeddings(text, modelfasttext):
    """
    FastText modelini kullanarak verilen metin için kelime gömülemeleri oluşturur.

    Args:
        text (str): Kelime gömülemeleri oluşturulacak metin.
        modelfasttext (fasttext.FastText._FastText): Yüklenen FastText modeli.

    Returns:
        list: Kelime gömülemelerinin (numpy dizileri) listesi.
    """
    bilinmeyen_word_say=0

    embeddings = []
    for word in text.split():
           
            # 'vocab' özelliği yoksa 'words' özelliğini kullanın
            if word in modelfasttext.words:
                embedding = modelfasttext.get_word_vector(word)
                embeddings.append(embedding)
            else:
                # Kelime benzerliği kullanımı
                 bilinmeyen_word_say=bilinmeyen_word_say+1
                 embedding = np.zeros(modelfasttext.get_dimension())
                 embeddings.append(embedding)

   # print(f"ilk makalede bilinmeyen kelime sayısı:{bilinmeyen_word_say}")
    return embeddings

#??MAKALELERİN KELİME VEKTOR DİZİLERİ
All_articles_vectors=[[]]


# her bir makalenin kelime vektörlerini oluştur
for i in range(len(preprocessed_train_abstracts)):
     first_train_abstract_embeddings = generate_embeddings(preprocessed_train_abstracts[i], modelfasttext)
    #!!!!!!!!!!!!!!!!!!! Bir makalenin titlenın  kelime ortalamasını bulmak

     mean_embedding = np.mean(first_train_abstract_embeddings, axis=0)
    # print("Kelimelerin Ortalaması:", mean_embedding)
     All_articles_vectors.append(mean_embedding)#bir makalenin ortalama vektörü

      #!!!!!!!!!!!!!!!!!!!!!!!




#!! giris yapan kullanıcın ilgialanlarinin ortalama vektörünün hesaplanması 
#giriş yapan kullanicini sifresi

config = {
                "apiKey": "AIzaSyCVwLgD-iJIkIqIRgOwbxBmx5vxKXsZP40",
                "authDomain": "makaleweb-a7bcf.firebaseapp.com",
                "databaseURL": "https://makaleweb-a7bcf-default-rtdb.europe-west1.firebasedatabase.app",
                "projectId": "makaleweb-a7bcf",
                "storageBucket": "makaleweb-a7bcf.appspot.com",
                "messagingSenderId": "156500986394",
                "appId": "1:156500986394:web:87b28a50afcdfb3815dfa3",
            
            };

firebase=pyrebase.initialize_app(config)
auth= firebase.auth()
database=firebase.database()


#veritababnından giren kullanıcını ilgi alanlarını çekme

# Veritabanından kullanıcılar öğesinin tum verilerini alma
tum_veriler = database.child("kullanicilar").get()

# Kullanicilarin ilgi alanlarini toplama
ilgi_alanlari = []#her article_onisleme_fatstext.py geldiğinde ilgi alanları sıfırlanır


# Kullanicilarin içine gir ve sifre degeri girenin sifresi olanların ilgiAlanlari degerini al
if tum_veriler is not None:
     for kullanici in tum_veriler.each():
        veri = kullanici.val()
        
        if veri is not None:
             databasede_girenin_sifre = veri.get('sifre')
             if (databasede_girenin_sifre=="55"):
                ilgi_alanlari=veri.get('ilgiAlanlari')
print(f"databasedeki girenin ilgi alanları{ilgi_alanlari}")   

#her bir kelimenin vektörlerini hesaplama ve ortalamasını alma



# Her bir kelimenin vektörünü hesaplayın
kelime_vektorleri =[]
for x in range(len(ilgi_alanlari)):
    kelime_vector = generate_embeddings(ilgi_alanlari[x], modelfasttext)
    kelime_vektorleri.append(kelime_vector)
# Farklı boyutlardaki vektörlerin ortalamasını hesaplayın
ortalama_vektor = np.zeros_like(kelime_vektorleri[0])  # İlk vektörün boyutunda bir sıfırlar dizisi oluşturun
for vektor in kelime_vektorleri:
    ortalama_vektor += vektor  # Her vektörü topla

ortalama_vektor /= len(kelime_vektorleri)  # Toplamı vektör sayısına bölererek ortalama al

print(f"ilgi alanı kelimelerinin ortalama vektörü: {ortalama_vektor}")

#niye iki boyutlu dizi çıktı ve cosine similarity


# Calculate cosine similarity  bu çıktı olayına bak
# Calculate cosine similarity
cosine_similarities = []
for article_vector in All_articles_vectors:
    article_vector = np.array(article_vector)  # Convert to 1-D array
    ortalama_vektor = np.array(ortalama_vektor).flatten()  # Convert to 1-D array
    cosine_similarity = cosine(article_vector, ortalama_vektor)
    cosine_similarities.append(cosine_similarity)

best_index = cosine_similarities.index(max(cosine_similarities))
print(f"Best article index: {best_index}")
print(f"Best cosine similarity: {max(cosine_similarities)}")


'''

#6   cosine similarity tamam
'''

# nltk_onislem.py den  preprocess_text fonksiyonunu alma
from nltk_onislem import  preprocess_text
from datasets import load_dataset
from fasttext_model import model
import numpy as np
from scipy.spatial.distance import cosine
#firebase ile ilgili işlemler için
import pyrebase
from pyrebase import initialize_app
#giren kullanıcını sifresini alma
from globaluser import giren_userpassword
#dataseti yükleme
dataset = load_dataset("taln-ls2n/inspec")
#cosine için scipy
from scipy.spatial.distance import cosine

#datasetteki tüm abstract temizle ÖN İŞLEME


# Train, validation ve test bölümlerini al
train_documents = dataset['train']
validation_documents = dataset['validation']
test_documents = dataset['test']

# Her bölümdeki her belge için abstract'i ön işlemden geçirme
preprocessed_train_abstracts = [preprocess_text(doc['title']) for doc in train_documents]
preprocessed_validation_abstracts = [preprocess_text(doc['title']) for doc in validation_documents]
preprocessed_test_abstracts = [preprocess_text(doc['title']) for doc in test_documents]

# Ön işlenmiş abstract'leri kontrol etme
print("Ön İşlenmiş Train Abstracts:")
print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
print("\nÖn İşlenmiş Validation Abstracts:")
print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
print("\nÖn İşlenmiş Test Abstracts:")
print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract






#fasttext modeli fasttext_model.py al

modelfasttext = model







def generate_embeddings(text, modelfasttext):
    """
    FastText modelini kullanarak verilen metin için kelime gömülemeleri oluşturur.

    Args:
        text (str): Kelime gömülemeleri oluşturulacak metin.
        modelfasttext (fasttext.FastText._FastText): Yüklenen FastText modeli.

    Returns:
        list: Kelime gömülemelerinin (numpy dizileri) listesi.
    """
    bilinmeyen_word_say=0

    embeddings = []
    for word in text.split():
           
            # 'vocab' özelliği yoksa 'words' özelliğini kullanın
            if word in modelfasttext.words:
                embedding = modelfasttext.get_word_vector(word)
                embeddings.append(embedding)
            else:
                # Kelime benzerliği kullanımı
                 bilinmeyen_word_say=bilinmeyen_word_say+1
                 embedding = np.zeros(modelfasttext.get_dimension())
                 embeddings.append(embedding)

   # print(f"ilk makalede bilinmeyen kelime sayısı:{bilinmeyen_word_say}")
    return embeddings

#??MAKALELERİN KELİME VEKTOR DİZİLERİ
All_articles_vectors=[]


# her bir makalenin kelime vektörlerini oluştur
for i in range(len(preprocessed_train_abstracts)):
     first_train_abstract_embeddings = generate_embeddings(preprocessed_train_abstracts[i], modelfasttext)
    #!!!!!!!!!!!!!!!!!!! Bir makalenin titlenın  kelime ortalamasını bulmak

     mean_embedding = np.mean(first_train_abstract_embeddings, axis=0)
    # print("Kelimelerin Ortalaması:", mean_embedding)
     All_articles_vectors.append(mean_embedding)#bir makalenin ortalama vektörü

      #!!!!!!!!!!!!!!!!!!!!!!!




#!! giris yapan kullanıcın ilgialanlarinin ortalama vektörünün hesaplanması 
#giriş yapan kullanicini sifresi

config = {
                "apiKey": "AIzaSyCVwLgD-iJIkIqIRgOwbxBmx5vxKXsZP40",
                "authDomain": "makaleweb-a7bcf.firebaseapp.com",
                "databaseURL": "https://makaleweb-a7bcf-default-rtdb.europe-west1.firebasedatabase.app",
                "projectId": "makaleweb-a7bcf",
                "storageBucket": "makaleweb-a7bcf.appspot.com",
                "messagingSenderId": "156500986394",
                "appId": "1:156500986394:web:87b28a50afcdfb3815dfa3",
            
            };

firebase=pyrebase.initialize_app(config)
auth= firebase.auth()
database=firebase.database()


#veritababnından giren kullanıcını ilgi alanlarını çekme

# Veritabanından kullanıcılar öğesinin tum verilerini alma
tum_veriler = database.child("kullanicilar").get()

# Kullanicilarin ilgi alanlarini toplama
ilgi_alanlari = []#her article_onisleme_fatstext.py geldiğinde ilgi alanları sıfırlanır


# Kullanicilarin içine gir ve sifre degeri girenin sifresi olanların ilgiAlanlari degerini al
if tum_veriler is not None:
     for kullanici in tum_veriler.each():
        veri = kullanici.val()
        
        if veri is not None:
             databasede_girenin_sifre = veri.get('sifre')
             if (databasede_girenin_sifre=="55"):
                ilgi_alanlari=veri.get('ilgiAlanlari')
print(f"databasedeki girenin ilgi alanları{ilgi_alanlari}")   

#her bir kelimenin vektörlerini hesaplama ve ortalamasını alma



# Her bir kelimenin vektörünü hesaplayın

#???  ilgi alanı kelime vektörlerini
from sklearn.decomposition import PCA

def generate_combined_vector(words, model, target_dim=300):
    """
    Belirli kelimeler için kelime gömülü vektörlerini oluşturur ve birleştirerek hedef boyuta (varsayılan olarak 300) indirger.

    Args:
        words (list): Kelimelerin listesi.
        model (fasttext.FastText._FastText): FastText modeli.
        target_dim (int, optional): Hedef boyut. Vektörlerin boyutunu hedef boyuta indirmek için kullanılır. Varsayılan olarak 300.

    Returns:
        numpy.ndarray: Birleştirilmiş ve hedef boyuta indirgenmiş vektör.
    """
    embeddings = []
    for word in words:
        if word in model.words:
            embedding = model.get_word_vector(word)
            embeddings.append(embedding)

    # Embeddinglerin boyutunu kontrol et
    if len(embeddings) == 0:
        return np.zeros(target_dim)  # Hiçbir kelime bulunamadıysa sıfırlar dizisi döndür

    # Vektörlerin boyutunu birleştirme
    combined_vector = np.mean(embeddings, axis=0)

    # Boyut indirme
    if len(combined_vector) != target_dim:
        pca = PCA(n_components=target_dim)
        combined_vector = pca.fit_transform([combined_vector])[0]

    return combined_vector





# Örnek kullanım:

birlestirilmis_vektor = generate_combined_vector(ilgi_alanlari, modelfasttext)
print("Birleştirilmiş Vektör:")
print(birlestirilmis_vektor)
print("Birleştirilmiş Vektörün Boyutu:", birlestirilmis_vektor.shape)



#??

#niye iki boyutlu dizi çıktı ve cosine similarity


# Calculate cosine similarity  bu çıktı olayına bak
# Calculate cosine similarity
cosine_similarities = []
for article_vector in All_articles_vectors:
    article_vector = np.array(article_vector)  # Convert to 1-D array(iki katına boyutlandırılmış vektör)
    birlestirilmis_vektor = np.array(birlestirilmis_vektor)# Convert to 1-D array
    cosine_similarity = cosine(article_vector, birlestirilmis_vektor)
    cosine_similarities.append(cosine_similarity)


#!! murat article vectors boyutunu 600 getirmeye çalış ve ilgi alanındaki herbir kelime içinmi vektor oluşturuluyor yoksa tüm kelimeler içinmi vektor oluşturuluyor

# Sort together by cosine similarity (descending order)
sorted_results = sorted(zip(cosine_similarities, train_documents), reverse=True)

# Get the top 5 most similar articles
top_5_articles = sorted_results[:5]

for similarity, article in top_5_articles:
    print(f"Cosine Similarity: {similarity:.4f}, Title: {article['abstract']}")


'''
#7 global users kullanıcıyı alma




def baslangic_onerilerifasttext():

        # nltk_onislem.py den  preprocess_text fonksiyonunu alma
    from Home.nltk_onislem import  preprocess_text
    from datasets import load_dataset
    from Home.fasttext_model import model
    import numpy as np
    from scipy.spatial.distance import cosine
    #firebase ile ilgili işlemler için
    import pyrebase
    from pyrebase import initialize_app
    #giren kullanıcını sifresini alma
    from Home.globaluser import giren_userpassword
     #dataseti yükleme
    from Home.global_dataset import my_dataset
    dataset = my_dataset
    #cosine için scipy
    from scipy.spatial.distance import cosine

    print(f"---------------------------------------------------{giren_userpassword}")

    #datasetteki tüm abstract temizle ÖN İŞLEME


    # Train, validation ve test bölümlerini al
    train_documents = dataset['train']
   


    validation_documents = dataset['validation']
    test_documents = dataset['test']

    # Her bölümdeki her belge için abstract'i ön işlemden geçirme
    preprocessed_train_abstracts = [preprocess_text(doc['title']) for doc in train_documents]
    preprocessed_validation_abstracts = [preprocess_text(doc['title']) for doc in validation_documents]
    preprocessed_test_abstracts = [preprocess_text(doc['title']) for doc in test_documents]

    # Ön işlenmiş abstract'leri kontrol etme
    print("Ön İşlenmiş Train Abstracts:")
    print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
    print("\nÖn İşlenmiş Validation Abstracts:")
    print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
    print("\nÖn İşlenmiş Test Abstracts:")
    print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract






    #fasttext modeli fasttext_model.py al

    modelfasttext = model







    def generate_embeddings(text, modelfasttext):
        """
        FastText modelini kullanarak verilen metin için kelime gömülemeleri oluşturur.

        Args:
            text (str): Kelime gömülemeleri oluşturulacak metin.
            modelfasttext (fasttext.FastText._FastText): Yüklenen FastText modeli.

        Returns:
            list: Kelime gömülemelerinin (numpy dizileri) listesi.
        """
        bilinmeyen_word_say=0

        embeddings = []
        for word in text.split():
            
                # 'vocab' özelliği yoksa 'words' özelliğini kullanın
                if word in modelfasttext.words:
                    embedding = modelfasttext.get_word_vector(word)
                    embeddings.append(embedding)
                else:
                    # Kelime benzerliği kullanımı
                    bilinmeyen_word_say=bilinmeyen_word_say+1
                    embedding = np.zeros(modelfasttext.get_dimension())
                    embeddings.append(embedding)

    # print(f"ilk makalede bilinmeyen kelime sayısı:{bilinmeyen_word_say}")
        return embeddings

    #??MAKALELERİN KELİME VEKTOR DİZİLERİ
    All_articles_vectors=[]


    # her bir makalenin kelime vektörlerini oluştur
    for i in range(len(preprocessed_train_abstracts)):
        first_train_abstract_embeddings = generate_embeddings(preprocessed_train_abstracts[i], modelfasttext)
        #!!!!!!!!!!!!!!!!!!! Bir makalenin titlenın  kelime ortalamasını bulmak

        mean_embedding = np.mean(first_train_abstract_embeddings, axis=0)
        # print("Kelimelerin Ortalaması:", mean_embedding)
        All_articles_vectors.append(mean_embedding)#bir makalenin ortalama vektörü

        #!!!!!!!!!!!!!!!!!!!!!!!




    #!! giris yapan kullanıcın ilgialanlarinin ortalama vektörünün hesaplanması 
    #giriş yapan kullanicini sifresi

    config = {
  "apiKey": "AIzaSyB5cPW2u4BlnZsyn5S8MJo4jqYom3fPOiw",
  "authDomain": "makeleweb2.firebaseapp.com",
  "databaseURL": "https://makeleweb2-default-rtdb.europe-west1.firebasedatabase.app",
  "projectId": "makeleweb2",
  "storageBucket": "makeleweb2.appspot.com",
  "messagingSenderId": "65251424283",
  "appId": "1:65251424283:web:16934ba965175be83760be",
 
                };

    firebase=pyrebase.initialize_app(config)
    auth= firebase.auth()
    database=firebase.database()


    #veritababnından giren kullanıcını ilgi alanlarını çekme

    # Veritabanından kullanıcılar öğesinin tum verilerini alma
    tum_veriler = database.child("kullanicilar").get()

    # Kullanicilarin ilgi alanlarini toplama
    ilgi_alanlari = []#her article_onisleme_fatstext.py geldiğinde ilgi alanları sıfırlanır
    #!! girenin sifresi normalde giren_userpassword
    giren_sifresi=giren_userpassword


    # Kullanicilarin içine gir ve sifre degeri girenin sifresi olanların ilgiAlanlari degerini al
    if tum_veriler is not None:
        for kullanici in tum_veriler.each():
            veri = kullanici.val()
            
            if veri is not None:
                databasede_girenin_sifre = veri.get('sifre')
                if (databasede_girenin_sifre==giren_sifresi):
                    ilgi_alanlari=veri.get('ilgiAlanlari')
    print(f"databasedeki girenin ilgi alanları{ilgi_alanlari}")   
    print(f"girenin passwordu:{giren_userpassword}")
    #her bir kelimenin vektörlerini hesaplama ve ortalamasını alma



    # Her bir kelimenin vektörünü hesaplayın

    #???  ilgi alanı kelime vektörlerini
    from sklearn.decomposition import PCA

    def generate_combined_vector(words, model, target_dim=300):
        """
        Belirli kelimeler için kelime gömülü vektörlerini oluşturur ve birleştirerek hedef boyuta (varsayılan olarak 300) indirger.

        Args:
            words (list): Kelimelerin listesi.
            model (fasttext.FastText._FastText): FastText modeli.
            target_dim (int, optional): Hedef boyut. Vektörlerin boyutunu hedef boyuta indirmek için kullanılır. Varsayılan olarak 300.

        Returns:
            numpy.ndarray: Birleştirilmiş ve hedef boyuta indirgenmiş vektör.
        """
        embeddings = []
        for word in words:
            if word in model.words:
                embedding = model.get_word_vector(word)
                embeddings.append(embedding)

        # Embeddinglerin boyutunu kontrol et
        if len(embeddings) == 0:
            return np.zeros(target_dim)  # Hiçbir kelime bulunamadıysa sıfırlar dizisi döndür

        # Vektörlerin boyutunu birleştirme
        combined_vector = np.mean(embeddings, axis=0)

        # Boyut indirme
        if len(combined_vector) != target_dim:
            pca = PCA(n_components=target_dim)
            combined_vector = pca.fit_transform([combined_vector])[0]

        return combined_vector





    # Örnek kullanım:

    birlestirilmis_vektor = generate_combined_vector(ilgi_alanlari, modelfasttext)
    print("Birleştirilmiş Vektör:")
    print(birlestirilmis_vektor)
    print("Birleştirilmiş Vektörün Boyutu:", birlestirilmis_vektor.shape)



    #??

    #niye iki boyutlu dizi çıktı ve cosine similarity


    # Calculate cosine similarity  bu çıktı olayına bak
    # Calculate cosine similarity
    cosine_similarities = []
    for article_vector in All_articles_vectors:
        article_vector = np.array(article_vector)  # Convert to 1-D array(iki katına boyutlandırılmış vektör)
        birlestirilmis_vektor = np.array(birlestirilmis_vektor)# Convert to 1-D array
        cosine_similarity = cosine(article_vector, birlestirilmis_vektor)
        cosine_similarities.append(cosine_similarity)


    #!! murat article vectors boyutunu 600 getirmeye çalış ve ilgi alanındaki herbir kelime içinmi vektor oluşturuluyor yoksa tüm kelimeler içinmi vektor oluşturuluyor

    # Sort together by cosine similarity (descending order)
    sorted_results = sorted(zip(cosine_similarities, train_documents), reverse=True)

    # Get the top 5 most similar articles
    top_5_articles = sorted_results[:5]



    #!!!!!!!!!!!!!!!!!!  bu kısmı kullanıcı arayuzune yansıtacaksın/ firebase bu kısmı tut  giriş idsi ile 
    def fasttext_baslangic_oneri():
        article_details = []
        for similarity, article in top_5_articles:
            similarity=similarity-0.25#öok sıfır vektörü var
            #print(f"Cosine Similarity: {similarity:.4f}, Abstract: {article['abstract']}")
            
            article_details.append(f"Cosine Similarity: {similarity:.4f}, Abstract: {article['abstract']}")
        return article_details

    print(fasttext_baslangic_oneri())

    #firebase ekleme kısmı
    # Firebase veritabanına ekleme
    # "oneri" adlı anahtar altında bir yapı oluştur
    oneri = {
        "id": giren_sifresi,
        "fasttext_baslangic_oneri": fasttext_baslangic_oneri()
    }

    # Veritabanında "oneri" adlı anahtar altında oluşturulan yapıyı gönder
    database.child("baslangic_oneri").child(giren_sifresi).set(oneri)







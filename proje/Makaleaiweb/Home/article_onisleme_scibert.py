def baslangic_oneriscibert():
    from Home.scibert_model import model
    from Home.scibert_model import tokenizer
    import torch
    import pyrebase
    import numpy as np
    from datasets import load_dataset
     #dataseti yükleme
    from Home.global_dataset import my_dataset
    dataset = my_dataset
    #cosine için scipy
    from scipy.spatial.distance import cosine
    from Home.nltk_onislem import  preprocess_text
    from Home.globaluser import giren_userpassword

    #??tum makalelerin vektoru
    All_articles_vectors=[]

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
    #print(preprocessed_train_abstracts[:5])  # İlk 5 ön işlenmiş train abstract
    print("\nÖn İşlenmiş Validation Abstracts:")
    #print(preprocessed_validation_abstracts[:5])  # İlk 5 ön işlenmiş validation abstract
    print("\nÖn İşlenmiş Test Abstracts:")
    #print(preprocessed_test_abstracts[:5])  # İlk 5 ön işlenmiş test abstract


    for i in range(len(preprocessed_train_abstracts)):
        #ön işlenmiş makale title
        # Metni tokenlere ayır
        text = preprocessed_train_abstracts[i]
        encoded_input = tokenizer(text, return_tensors="pt")

            # Modelden vektör temsilini elde et
        with torch.no_grad():
            output = model(**encoded_input)
            vectors = output.last_hidden_state[:, 0, :] # İlk kelimenin vektör temsili
            All_articles_vectors.append(vectors)
           # print(vectors)
           # print("makale boyut")
            #print(vectors.shape)












    #olursa devam için oluşturulmuş mantıksal kod
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
 
    }

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

    print(f"\tdatabasedeki girenin ilgi alanları{ilgi_alanlari}")
    print(f"\tgirenin passwordu:{giren_sifresi}")


    ortalama_vektor = torch.zeros(size=[1, vectors.shape[1]])  # Ortalama vektör için boş bir tensor oluştur

    for ilgialani in ilgi_alanlari:
        # Metni tokenlere ayır
        text = ilgialani
        encoded_input = tokenizer(text, return_tensors="pt")

        # Modelden vektör temsilini elde et
        with torch.no_grad():
            output = model(**encoded_input)
            vectors = output.last_hidden_state[:, 0, :] # kelimenin vektör temsili 
        
        # Ortalama vektöre ekle
        ortalama_vektor += vectors 
        print(f"{ilgialani} için vektör temsili:")
        print(vectors)

    # Ortalama vektörü tüm ilgi alanları sayısına bölerek hesaplayın
    ortalama_vektor /= len(ilgi_alanlari)

    # Ortalama vektörü yazdırın
    print("Ortalama Vektör:")
    print(ortalama_vektor)
    print("ortalama vektor boyutu:")
    print(ortalama_vektor.shape)


    #!! benzerlik  için cosine similarity hesaplama
    # Calculate cosine similarity
    cosine_similarities = []
    for article_vector in All_articles_vectors:
        article_vector = np.array(article_vector).flatten()  # Convert to 1-D array(iki katına boyutlandırılmış vektör)
        birlestirilmis_vektor = np.array(ortalama_vektor).flatten()# Convert to 1-D array
        cosine_similarity = cosine(article_vector, birlestirilmis_vektor)
        cosine_similarities.append(cosine_similarity)


    #!! murat article vectors boyutunu 600 getirmeye çalış ve ilgi alanındaki herbir kelime içinmi vektor oluşturuluyor yoksa tüm kelimeler içinmi vektor oluşturuluyor

    # Sort together by cosine similarity (descending order)
    sorted_results = sorted(zip(cosine_similarities, train_documents), reverse=True)

    # Get the top 5 most similar articles
    top_5_articles = sorted_results[:5]
    #yazdırma
    def scibert_baslangiconerifonk():
        article_details = []
        for similarity, article in top_5_articles:
            article_details.append(f"Cosine Similarity: {similarity:.4f}, Abstract: {article['abstract']}")
        return article_details

    print(scibert_baslangiconerifonk())
    #firebase ekleme kısmı
    # Firebase veritabanına ekleme
    # "oneri" adlı anahtar altında bir yapı oluştur
    oneri = {
        "id": giren_sifresi,
        "scibert_baslangic_oneri": scibert_baslangiconerifonk()
    }

    # Veritabanında "oneri" adlı anahtar altında oluşturulan yapıyı gönder
    database.child("baslangic_oneri_scibert").child(giren_sifresi).set(oneri)


































 
   
'''

from scibert_model import model
from scibert_model import tokenizer
import torch
import pyrebase
import numpy as np
from datasets import load_dataset
#dataseti yükleme
dataset = load_dataset("taln-ls2n/inspec")
#cosine için scipy
from scipy.spatial.distance import cosine
from nltk_onislem import  preprocess_text
from globaluser import giren_userpassword

#??tum makalelerin vektoru
All_articles_vectors=[]

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


# İlgili belgelerin özetlerini almak için for döngüsü
for preprocessed_abstracts in [preprocessed_train_abstracts, preprocessed_validation_abstracts, preprocessed_test_abstracts]:
    for abstract in preprocessed_abstracts:
        # Her bir belgedeki kelimelerin vektörlerini tutacak bir liste oluşturun
        word_vectors = []
        # Her kelimeyi tokenleştirmek ve vektörlerini almak için for döngüsü
        for word in abstract.split():
            # Metni tokenlere ayır
            encoded_input = tokenizer(word, return_tensors="pt")
            # Modelden vektör temsilini al
            with torch.no_grad():
                output = model(**encoded_input)
                word_vector = output.last_hidden_state[:, 0, :] # Kelimenin vektör temsili
                word_vectors.append(word_vector)
        # Her belgedeki kelime vektörlerinin ortalamasını alın
        document_vector = torch.mean(torch.stack(word_vectors), dim=0)
        All_articles_vectors.append(document_vector)











#olursa devam için oluşturulmuş mantıksal kod
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
 
}

firebase=pyrebase.initialize_app(config)
auth= firebase.auth()
database=firebase.database()

#veritababnından giren kullanıcını ilgi alanlarını çekme

# Veritabanından kullanıcılar öğesinin tum verilerini alma
tum_veriler = database.child("kullanicilar").get()

# Kullanicilarin ilgi alanlarini toplama
ilgi_alanlari = []#her article_onisleme_fatstext.py geldiğinde ilgi alanları sıfırlanır
#!! girenin sifresi normalde giren_userpassword
giren_sifresi="12"

# Kullanicilarin içine gir ve sifre degeri girenin sifresi olanların ilgiAlanlari degerini al
if tum_veriler is not None:
    for kullanici in tum_veriler.each():
        veri = kullanici.val()
        
        if veri is not None:
            databasede_girenin_sifre = veri.get('sifre')
            if (databasede_girenin_sifre==giren_sifresi):
                ilgi_alanlari=veri.get('ilgiAlanlari')

print(f"\tdatabasedeki girenin ilgi alanları{ilgi_alanlari}")
print(f"\tgirenin passwordu:{giren_sifresi}")


ortalama_vektor = torch.zeros(size=[1, 768])  # Ortalama vektör için boş bir tensor oluştur


for ilgialani in ilgi_alanlari:
   # Metni tokenlere ayır
   text = ilgialani
   encoded_input = tokenizer(text, return_tensors="pt")

   # Modelden vektör temsilini elde et
   with torch.no_grad():
    output = model(**encoded_input)
    vectors = output.last_hidden_state[:, 0, :] # kelimenin vektör temsili 
  
   # Ortalama vektöre ekle
   ortalama_vektor += vectors 
   print(f"{ilgialani} için vektör temsili:")
   print(vectors)

# Ortalama vektörü tüm ilgi alanları sayısına bölerek hesaplayın
ortalama_vektor /= len(ilgi_alanlari)

# Ortalama vektörü yazdırın
print("Ortalama Vektör:")
print(ortalama_vektor)
print("ortalama vektor boyutu:")
print(ortalama_vektor.shape)


#!! benzerlik  için cosine similarity hesaplama
# Calculate cosine similarity
cosine_similarities = []
for article_vector in All_articles_vectors:
    article_vector = np.array(article_vector).flatten()  # Convert to 1-D array(iki katına boyutlandırılmış vektör)
    birlestirilmis_vektor = np.array(ortalama_vektor).flatten()# Convert to 1-D array
    cosine_similarity = cosine(article_vector, birlestirilmis_vektor)
    cosine_similarities.append(cosine_similarity)


#!! murat article vectors boyutunu 600 getirmeye çalış ve ilgi alanındaki herbir kelime içinmi vektor oluşturuluyor yoksa tüm kelimeler içinmi vektor oluşturuluyor

# Sort together by cosine similarity (descending order)
sorted_results = sorted(zip(cosine_similarities, train_documents), reverse=True)

# Get the top 5 most similar articles
top_5_articles = sorted_results[:5]
#yazdırma
for similarity, article in top_5_articles:
    print(f"Cosine Similarity: {similarity:.4f}, abstract: {article['abstract']}")


'''










































































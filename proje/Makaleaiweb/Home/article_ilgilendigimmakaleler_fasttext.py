

def ilgilendigimmakale_onerifasttext():
    import torch
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
    dataset = load_dataset("taln-ls2n/inspec")
    #cosine için scipy
    from scipy.spatial.distance import cosine
  #!! girenin sifresi normalde giren_userpassword
    giren_sifresi=giren_userpassword

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


    #????????veritababnından giren kullanıcını  ilgilendiği makalelerin ortalama  vektorunu oluşturma
  
    #veritababnından giren kullanıcını ilgilendiği makaleler varsa onları alma
#ilgilimakale
#72
#abstract_scibert


    # Veritabanından ilgilimakale/72 düğümündeki verileri alın
    user_id = giren_sifresi  # Kullanıcının ID'si
    ilgilimakale_ref = f"ilgilimakale/{user_id}/abstract_fasttext"  # Veri yolu
    ilgilimakale_verileri = database.child(ilgilimakale_ref).get()
    All_abstract=[]#!!tum abstarct tutuyorum
    All_ilgilendigim_articles=[]#!!tum ilgilendigim makalelerin vektorü
    # Verileri kontrol edin ve işleyin
    if ilgilimakale_verileri.each():
        for veri in ilgilimakale_verileri.each():
     #       print(veri.key(), veri.val())  # Düğüm adını ve değerini yazdırın
             abstract=None
             if "Abstract" in veri.val():
                  abstract_start = veri.val().index("Abstract") + 8
                  abstract = veri.val()[abstract_start:]  # "Abstract"den 8 konum sonra özeti çıkarın
             else:
                 abstract = None 
             All_abstract.append(abstract)
    else:
        print("Kullanıcının ilgilendiği makale bulunamadı.")

 # her bir makalenin kelime vektörlerini oluştur
    for i in range(len(All_abstract)):
        first_train_abstract_embeddings = generate_embeddings(All_abstract[i], modelfasttext)
        #!!!!!!!!!!!!!!!!!!! Bir makalenin ABSTRACT  kelime ortalamasını bulmak

        mean_embedding = np.mean(first_train_abstract_embeddings, axis=0)
        # print("Kelimelerin Ortalaması:", mean_embedding)
        All_ilgilendigim_articles.append(mean_embedding)#bir makalenin  abstract ortalama vektörü

   
  # Ortalama vektöR İLGİLENDİĞİM MAKALELERİNhesapla
    ortalama_ilgiarticles_vektor = np.mean(All_ilgilendigim_articles, axis=0)

# Ortalama vektörü yazdır
    print(f"ortalama_ilgiarticles_vektor{ortalama_ilgiarticles_vektor}ve boyu{ortalama_ilgiarticles_vektor.shape}")









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



    #??birlestirilmis vektor ile ortalama_ilgiarticles_vektorunun  ortalaması 
     # `ortalama_ilgiarticles_vektor`'ü PyTorch tensorüne dönüştür
    ortalama_ilgiarticles_vektor_tensor = torch.tensor(ortalama_ilgiarticles_vektor)

# Her bir satırın ortalama vektörünü hesapla ve PyTorch tensorüne dönüştür
    ortalama_vektor_tensor = torch.tensor(birlestirilmis_vektor)

# İki tensorü topla ve ortalamasını al
    ortalama_ilgialan_ilgimakale_tensor = (ortalama_ilgiarticles_vektor_tensor + ortalama_vektor_tensor) / 2


    #niye iki boyutlu dizi çıktı ve cosine similarity


    # Calculate cosine similarity  bu çıktı olayına bak
    # Calculate cosine similarity
    cosine_similarities = []
    for article_vector in All_articles_vectors:
        article_vector = np.array(article_vector)  # Convert to 1-D array(iki katına boyutlandırılmış vektör)
        birlestirilmis_vektor = np.array(ortalama_ilgialan_ilgimakale_tensor)# Convert to 1-D array
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
            #print(f"Cosine Similarity: {similarity:.4f}, Abstract: {article['abstract']}")
           similarity=similarity-0.10#daha az sıfır vektörü var
           article_details.append(f"Cosine Similarity: {similarity:.4f}, Abstract: {article['abstract']}")
        return article_details

    print(fasttext_baslangic_oneri())

 
      #firebase ekleme kısmı
    # Firebase veritabanına ekleme
    # "oneri" adlı anahtar altında bir yapı oluştur
    oneri = {
        "id": giren_sifresi,
        "fasttext_ilgilendigiarticles_oneri": fasttext_baslangic_oneri()
    }

    # Veritabanında "oneri" adlı anahtar altında oluşturulan yapıyı gönder
    database.child("ilgilendigi_makaleye_fasttextonerisi").child(giren_sifresi).set(oneri)







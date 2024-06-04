def ilgilendigimmakaleler_scibert(onerilen_articles):
    from scibert_model import model
    from scibert_model import tokenizer
    import torch
    import pyrebase
    import numpy as np
    from datasets import load_dataset
     #dataseti yükleme
    from global_dataset import my_dataset
    dataset = my_dataset
    #cosine için scipy
    from scipy.spatial.distance import cosine
    from nltk_onislem import  preprocess_text
    from globaluser import giren_userpassword
       #!! girenin sifresi normalde giren_userpassword
    giren_sifresi="kemal1"
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
            #print(vectors)
          #  print("makale boyut")
           # print(vectors.shape)












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
    
    #veritababnından giren kullanıcını ilgilendiği makaleler varsa onları alma
#ilgilimakale
#72
#abstract_scibert


    # Veritabanından ilgilimakale/72 düğümündeki verileri alın
    user_id = giren_sifresi  # Kullanıcının ID'si
    ilgilimakale_ref = f"ilgilimakale/{user_id}/abstract_scibert"  # Veri yolu
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


    for i in range(len(All_abstract)):
        #ön işlenmiş makale title
        # Metni tokenlere ayır
        text = All_abstract[i]
        encoded_input = tokenizer(text, return_tensors="pt")

            # Modelden vektör temsilini elde et
        with torch.no_grad():
            output = model(**encoded_input)
            vectors = output.last_hidden_state[:, 0, :] # İlk kelimenin vektör temsili
            All_ilgilendigim_articles.append(vectors)
           # print(vectors)
           # print("makale boyut")
            #print(vectors.shape)

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
    #??ORTALAMA VEKTOR İLE  ortalama_ilgiarticles_vektor ORTALAMASINI ALMAK
    # `ortalama_ilgiarticles_vektor`'ü PyTorch tensorüne dönüştür
    ortalama_ilgiarticles_vektor_tensor = torch.tensor(ortalama_ilgiarticles_vektor)



####################!! GELEN PARAMETRE FONKSİYONA onerilen_articles
    # Önerilen makalelerin vektörlerini hesaplayın
    onerilen_makale_vektorleri = []
    for makale_metni in onerilen_articles:
        encoded_input = tokenizer(makale_metni, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded_input)
            vektor = output.last_hidden_state[:, 0, :]  # İlk kelimenin vektör temsili
            onerilen_makale_vektorleri.append(vektor)

    # Önerilen makale vektörlerinin ortalamasını hesapla
    ortalama_onerilen_makale_vektor = torch.mean(torch.stack(onerilen_makale_vektorleri), dim=0)





# Her bir satırın ortalama vektörünü hesapla ve PyTorch tensorüne dönüştür
    ortalama_vektor_tensor = torch.tensor((ortalama_vektor+ortalama_onerilen_makale_vektor)/2)


# İki tensorü topla ve ortalamasını al
    ortalama_ilgialan_ilgimakale_tensor = (ortalama_ilgiarticles_vektor_tensor + ortalama_vektor_tensor) / 2

    #!! benzerlik  için cosine similarity hesaplama
    # Calculate cosine similarity
    cosine_similarities = []
    for article_vector in All_articles_vectors:
        article_vector = np.array(article_vector).flatten()  # Convert to 1-D array(iki katına boyutlandırılmış vektör)
        birlestirilmis_vektor = np.array(ortalama_ilgialan_ilgimakale_tensor).flatten()# Convert to 1-D array
        cosine_similarity = cosine(article_vector, birlestirilmis_vektor)
        cosine_similarities.append(cosine_similarity)


    #!! murat article vectors boyutunu 600 getirmeye çalış ve ilgi alanındaki herbir kelime içinmi vektor oluşturuluyor yoksa tüm kelimeler içinmi vektor oluşturuluyor
   #ilk 5 sonucu değilde 30 sonucu getirsin
    # Sort together by cosine similarity (descending order)
    sorted_results = sorted(zip(cosine_similarities, train_documents), reverse=True)

    # en çok benzeyen 50 makaleyi al ve getir
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
        "scibert_tum_oneri": scibert_baslangiconerifonk()
    }

    # Veritabanında "oneri" adlı anahtar altında oluşturulan yapıyı gönder
    database.child("tum_scibertonerisi").child(giren_sifresi).set(oneri)



# Önerilen makalelerin metinlerini bir listeye yerleştirin
veri = [
    "In the last decade we have both monitored with great interest the ratio of female to male computer science majors at our respective institutions.",
    "With each entering class, we think: 'Surely, now is the time when the numbers will become more balanced.' Logic tells us that this must eventually happen, because the opportunities in computing are simply too attractive for an entire segment of our population to routinely pass up. But each year we are again disappointed in the number of women students, as they continue to be woefully under-represented among computer science majors."
]
ilgilendigimmakaleler_scibert(veri)
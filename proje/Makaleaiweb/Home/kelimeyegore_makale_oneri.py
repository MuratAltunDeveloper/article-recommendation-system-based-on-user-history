#views.py içinde bir fonksiyon ile bunu çağır  ve search worde göre makaleleri getir
def searchword_oneri():
    import torch
    import pyrebase
    import numpy as np
    from datasets import load_dataset
    #dataseti yükleme
    dataset = load_dataset("taln-ls2n/inspec")
    #global şifre
    from Home.globaluser import giren_userpassword
    #!! girenin sifresi normalde giren_userpassword
    giren_sifresi=giren_userpassword
    #!!!arama kelimeside burada
    arama_kelimesi="women"
    #datasetin train kısmını alıyorum
    train_documents = dataset['train']
    #anahtar kelimelerini aldım
    preprocessed_train_abstracts = [(doc['keyphrases']) for doc in train_documents]
    oneri_makaleler=[]
    print(preprocessed_train_abstracts)
    #kelime dizide varmı kontrol etme
    for x in range(len(preprocessed_train_abstracts)):
        if arama_kelimesi in preprocessed_train_abstracts[x]:
            searchword_inarticle=train_documents[x]['abstract']
            print(f"{searchword_inarticle} anahtar kelimeler listesinde bulunuyor.")
            oneri_makaleler.append(searchword_inarticle)
            print("********************---------------------***********************")
        else:
            pass

    return oneri_makaleler

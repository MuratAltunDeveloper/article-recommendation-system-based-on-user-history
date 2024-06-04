'''
#https://drive.google.com/drive/u/0/folders/1lMHw48EvQy3V0fm8A9Woi3Y-4aane7Nv  yazlab12 drive
#django 
from django.shortcuts import render
from django.http.response import HttpResponse
import pyrebase
import tkinter as tk
from tkinter import messagebox
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
import logging
logger = logging.getLogger(__name__)#django print yerine loging kullanır



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


giris_name="yok"
giris_sifre="000"


# Create your views here.
def kayit(request):
    return render(request,"kayit.html")
def users(request):
    return render(request,"kullanicilar.html")
def home(request):
    return render(request,"homepage.html")

# Giriş sayfası görünümü
# ... Diğer fonksiyonlar ...

def giris(request):
    if request.method == 'POST':
        giris_adi = request.POST['username']
        giris_sifre = request.POST['password']
        print(f'{giris_adi} adlı ve sifresi:{giris_sifre}')        
    
    return render(request, 'giris.html')

# Hakkımızda sayfası görünümü
def hakkimizda(request):
    return render(request, "hakkimizda.html")

def updateuser(request):
    return render(request,"Profil_Goruntuleme_ve_Duzenleme.html")




'''


####   222222222222222   kısım  giriste kullanıcı ve  sifre alınmış

'''
#https://drive.google.com/drive/u/0/folders/1lMHw48EvQy3V0fm8A9Woi3Y-4aane7Nv  yazlab12 drive
#django 
from django.shortcuts import render
from django.http.response import HttpResponse
import pyrebase
import tkinter as tk
from tkinter import messagebox
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
import logging
logger = logging.getLogger(__name__)#django print yerine loging kullanır



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


giris_name="yok"
giris_sifre="000"


# Create your views here.
def kayit(request):
    return render(request,"kayit.html")
def users(request):
    return render(request,"kullanicilar.html")
def home(request):
    return render(request,"homepage.html")

# Giriş sayfası görünümü
# ... Diğer fonksiyonlar ...


def giris(request):
    if request.method == 'POST':
        giris_adi = request.POST['username']
        giris_sifre = request.POST['password']
        print(f'{giris_adi} adlı ve sifresi:{giris_sifre}')        
    
     # Tüm kullanıcı verilerini çekmek için
    tum_veriler = database.child("kullanicilar").get()
    
     # Veritabanından gelen verileri işle
    for kullanici in tum_veriler.each():
        # Her bir kullanıcı için veriye eriş
        veri = kullanici.val()
        print(veri)  # Her bir kullanıcı verisini yazdır
        databasede_kullanici_adi = veri['isim']
        databasede_sifre = veri['sifre']
        databasedevarmi=False
        if((databasede_kullanici_adi==giris_adi)and(databasede_sifre==giris_sifre)):
            databasedevarmi=True
            return redirect('users')

    
    if(databasedevarmi==False):
        #database yoksa
        messages.error(request, 'Hatalı kullanıcı adı veya şifre.')
        return redirect('giris')  # Giriş sayfasına geri yönlendir

    else:
        return render(request, 'giris.html')







# Hakkımızda sayfası görünümü
def hakkimizda(request):
    return render(request, "hakkimizda.html")

def updateuser(request):
    return render(request,"Profil_Goruntuleme_ve_Duzenleme.html")






'''




# 33333333333         giren kullanıcı firebasede varmı kontrol


#https://drive.google.com/drive/u/0/folders/1lMHw48EvQy3V0fm8A9Woi3Y-4aane7Nv  yazlab12 drive
#django 
from django.shortcuts import render
from django.http.response import HttpResponse
import pyrebase
import tkinter as tk
from tkinter import messagebox
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
import logging
from Home.globaluser import get_kullanici_password
from Home.article_onisleme_fasttext import baslangic_onerilerifasttext
from Home.article_onisleme_scibert import baslangic_oneriscibert
from Home.article_ilgilendigimmakaleler_fasttext import ilgilendigimmakale_onerifasttext
from Home.article_ilgilendigimmakaleler_scibert import ilgilendigimmakaleler_scibert
from Home.kelimeyegore_makale_oneri import searchword_oneri
from Home.global_dataset import my_dataset
from Home.article_tumisterleregore_fasttext import tumoneri_fasttext
from Home.article_tumisterleregore_scibert import tumoneri_scibert
logger = logging.getLogger(__name__)#django print yerine loging kullanır



#firestore kısmı 
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


giris_name="yok"
giris_sifre="000"


# Create your views here.
def kayit(request):
    return render(request,"kayit.html")
def users(request):
    
    return render(request,"kullanicilar.html")
def home(request):
    #sayfaya girerken bir kez dataseti yukleme
    veri=my_dataset
    return render(request,"homepage.html")

# Giriş sayfası görünümü
def giris(request):
    if request.method == 'POST':
        if 'username' in request.POST and 'password' in request.POST:
            giris_adi = request.POST['username']
            giris_sifre = request.POST['password']
            print(f'{giris_adi} adlı ve sifresi:{giris_sifre}')
            tum_veriler = database.child("kullanicilar").get()
    
            databasedevarmi = False
            if tum_veriler is not None:
                for kullanici in tum_veriler.each():
                    veri = kullanici.val()
                    print(veri)  # Her bir kullanıcı verisini yazdır
                    if veri is not None:
                        databasede_kullanici_adi = veri.get('isim')
                        databasede_sifre = veri.get('sifre')
                        if databasede_kullanici_adi == giris_adi and databasede_sifre == giris_sifre:
                            databasedevarmi = True
                             # Örnek şifre verisi
                            password_data = {
                               'sifrehtml': giris_sifre,
                               'adhtml':giris_adi
                            }
                            get_kullanici_password(giris_sifre)
                            baslangic_onerilerifasttext()
                            baslangic_oneriscibert()
                            return render(request, 'kullanicilar.html',context=password_data)#!path giris olarak kalıyor

            if not databasedevarmi:
                messages.error(request, 'Hatalı kullanıcı adı veya şifre.')
                return render(request, 'giris.html')  # Giriş sayfasına geri yönlendir

    else:
        return render(request, 'giris.html')


# Hakkımızda sayfası görünümü
def hakkimizda(request):
    return render(request, "hakkimizda.html")

def updateuser(request, girenpassword):
    print("------->"+girenpassword)
    context={
       'girensifre': girenpassword, 
    }
    return render(request,"Profil_Goruntuleme_ve_Duzenleme.html",context)

def ilgilendigimakaleleri_goster(request,girenpassword):
    context={
        'girensifre': girenpassword,
    }
    #burada bir pyde ilgi alanına göre makaleler olacak onu çalıştır
    ilgilendigimmakale_onerifasttext()
    ilgilendigimmakaleler_scibert()
    return render(request,"ilgi_alanina_goremakaleler.html",context)



def search_view(request):
    kelime = request.GET.get('kelime')  # Arama kelimesini al
    sifre = request.GET.get('sifre')  # Şifreyi al
    # Burada arama kelimesiyle ilgili işlemleri yapabilirsiniz
    onerilen_articles=searchword_oneri()
    return render(request, 'search_results.html', {'kelime': kelime,'girensifre':sifre,'onerilen_articles':onerilen_articles})

def oneridegerlendirme(request, girenpassword):
    context={
       'girensifre': girenpassword, 
    }
    return render(request,'oneridegerlendirme.html',context)

def geneloneri(request, girenpassword,kelime):
    onerilen_articles=searchword_oneri()
    context={
       'kelime':kelime,
       'girensifre': girenpassword, 
       'oneriler':onerilen_articles
    }


    tumoneri_scibert(onerilen_articles)
    tumoneri_fasttext(onerilen_articles)
    return render(request,'oneridegerlendirme.html',context)






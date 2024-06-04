// Firebase yapılandırma bilgilerini buraya ekleyin
var firebaseConfig = {

    "apiKey": "AIzaSyB5cPW2u4BlnZsyn5S8MJo4jqYom3fPOiw",
    "authDomain": "makeleweb2.firebaseapp.com",
    "databaseURL": "https://makeleweb2-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "makeleweb2",
    "storageBucket": "makeleweb2.appspot.com",
    "messagingSenderId": "65251424283",
    "appId": "1:65251424283:web:16934ba965175be83760be",
   
    
    
  };
  
  // Firebase'i başlatın
  firebase.initializeApp(firebaseConfig);
  
  // Realtime Database referansını alın
  var db = firebase.database();
  
  document.getElementById("kayitFormu").addEventListener("submit", function(event) {
    event.preventDefault(); // Formun otomatik olarak gönderilmesini engelle

    // Formdan verileri al
    var isim = document.getElementById("isim").value;
    var sifre = document.getElementById("sifre").value;
    var dogumTarihi = document.getElementById("dogumTarihi").value;
    var cinsiyet = document.getElementById("cinsiyet").value;
    var ilgiAlanlari = Array.from(document.getElementById("ilgiAlanlari").selectedOptions).map(option => option.value);

    // Veriyi Realtime Database'e ekle
    var newUserData = {
        isim: isim,
        sifre: sifre,
        dogumTarihi: dogumTarihi,
        cinsiyet: cinsiyet,
        ilgiAlanlari: ilgiAlanlari
    };

    // Veritabanındaki mevcut kullanıcı sayısını al
    db.ref('kullanicilar').once('value', function(snapshot) {
        var userCount = snapshot.numChildren();

        // 'kullanicilar' düğümüne yeni kullanıcı verisini ekle
        db.ref('kullanicilar').child(userCount + 1).set(newUserData)
            .then(function() {
                console.log("Kullanıcı başarıyla eklendi!");
                // Başka bir işlem yapabilirsiniz, örneğin kullanıcıyı başka bir sayfaya yönlendirebilirsiniz.
                window.location.href = "http://127.0.0.1:8000/";

            })
            .catch(function(error) {
                console.error("Veri eklenirken bir hata oluştu:", error);
                // Hata durumunda gerekli işlemleri yapabilirsiniz
                alert("Veri kaydedilemedi. Bir hata oluştu.");
            });
    });
});

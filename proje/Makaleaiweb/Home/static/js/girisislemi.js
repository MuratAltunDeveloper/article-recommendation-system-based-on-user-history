/*
// Firebase yapılandırma bilgilerini buraya ekleyin
var firebaseConfig = {

    "apiKey": "AIzaSyCVwLgD-iJIkIqIRgOwbxBmx5vxKXsZP40",
    "authDomain": "makaleweb-a7bcf.firebaseapp.com",
    "databaseURL": "https://makaleweb-a7bcf-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "makaleweb-a7bcf",
    "storageBucket": "makaleweb-a7bcf.appspot.com",
    "messagingSenderId": "156500986394",
    "appId": "1:156500986394:web:87b28a50afcdfb3815dfa3",


};

// Firebase'i başlatın
firebase.initializeApp(firebaseConfig);

// Realtime Database referansını alın
var db = firebase.database();

//realtime  kullanicilarda bu veri varmı kontrol etme

document.getElementById("loginForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Formun otomatik olarak gönderilmesini engelle

    var username=document.getElementById("username").value;
    var password = document.getElementById("password").value;
   
    var kullanicilarRef = db.ref('kullanicilar');
 // Kullanıcılar referansındaki verileri dinle
//kontrol kullanıcılara sayfasına yoneldimi
var isUserFound = false; // Başlangıçta bayrak false



 kullanicilarRef.once('value', function(snapshot) {
    // Her bir kullanıcı verisini kontrol et
    snapshot.forEach(function(childSnapshot) {
        var userData = childSnapshot.val(); // Kullanıcı verisi
        var userIsim = userData.isim; // Kullanıcı adı
        var userSifre = userData.sifre; // Kullanıcı şifresi


       
        // Kullanıcı adı ve şifre eşleşiyor mu kontrol et
        if (userIsim === username && userSifre === password) {
            console.log("Eşleşen kullanıcı adı ve şifre.");
           isUserFound = true; // Bayrağı true olarak ayarlayın
            // Giriş başarılı olduğunda başka bir sayfaya yönlendirme yapabilirsiniz
            window.location.href = "http://127.0.0.1:8000/users";
            return; // Eşleşme bulunduğunda fonksiyondan çık
        }
    });


    // Döngü tamamlandıktan sonra
if (isUserFound) {
    console.log("İşlem tamamlandı.");
    // Eşleşme bulunamadıysa
    
  } else {
    console.log("Eşleşen kullanıcı bulunamadı."); 
    alert("Hatalı kullanıcı adı veya şifre.");
    window.location.href = "http://127.0.0.1:8000/";
 
  }
    

});






});
*/



///222  kod kısmı 
// Firebase yapılandırma bilgilerini buraya ekleyin
/*
var firebaseConfig = {
    "apiKey": "AIzaSyCVwLgD-iJIkIqIRgOwbxBmx5vxKXsZP40",
    "authDomain": "makaleweb-a7bcf.firebaseapp.com",
    "databaseURL": "https://makaleweb-a7bcf-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "makaleweb-a7bcf",
    "storageBucket": "makaleweb-a7bcf.appspot.com",
    "messagingSenderId": "156500986394",
    "appId": "1:156500986394:web:87b28a50afcdfb3815dfa3",
};

// Firebase'i başlatın
firebase.initializeApp(firebaseConfig);

// Realtime Database referansını alın
var db = firebase.database();

// Form gönderildiğinde
document.getElementById("loginForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Formun otomatik olarak gönderilmesini engelle

    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;

    // Realtime Database'de kullanıcılar referansı
    var kullanicilarRef = db.ref('kullanicilar');

    // Kullanıcılar referansındaki verileri dinle
    kullanicilarRef.once('value', function(snapshot) {
        // Her bir kullanıcı verisini kontrol et
        snapshot.forEach(function(childSnapshot) {
            var userData = childSnapshot.val(); // Kullanıcı verisi
            var userIsim = userData.isim; // Kullanıcı adı
            var userSifre = userData.sifre; // Kullanıcı şifresi

            // Kullanıcı adı ve şifre eşleşiyor mu kontrol et
            if (userIsim === username && userSifre === password) {
                console.log("Eşleşen kullanıcı adı ve şifre.");
                // Başarılı giriş durumunda, formu gönder ve işlemi tamamla
                document.getElementById("loginForm").submit();
                return; // Eşleşme bulunduğunda fonksiyondan çık
            }
        });

        // Eğer eşleşme bulunamadıysa, hata mesajı göster
        alert("Hatalı kullanıcı adı veya şifre.");
    });
});
*/
global giren_userpassword#global
giren_userpassword=0

def get_kullanici_password(sifre):
    """Giriş yapan kullanıcının kullanıcı adını döndürür."""
    if sifre is not None:
      global giren_userpassword
      giren_userpassword=sifre  
      print(f"-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-{giren_userpassword}")
# ... (Kalan kod)

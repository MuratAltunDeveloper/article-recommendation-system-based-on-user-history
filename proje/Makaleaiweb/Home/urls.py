from django.urls import path
from . import views

#http://127.0.0.1:8000/           index
#http://127.0.0.1:8000/index      index
#http://127.0.0.1:8000/users      userpage
#
urlpatterns = [
    path("",views.home, name="home"),
    path("kayitol/",views.kayit,name="kayitol"),
    path("users/", views.users,name="USERS"),
    path("giris/", views.giris, name="giris"),
    path("hakkimizda/", views.hakkimizda, name="hakkimizda"),
    path("users/updateusers/<str:girenpassword>/",views.updateuser,name="users_profile_update"),
    path("ilgilendigiarticles/<str:girenpassword>/",views.ilgilendigimakaleleri_goster,name="users_ilgilendigi_makale"),
    path('arama/', views.search_view, name='search'),
    path('oneridegerlendirme/<str:girenpassword>/',views.oneridegerlendirme,name='users_oneridegerlendirme'),
    path('geneloneri/<str:girenpassword>/<str:kelime>/', views.geneloneri, name='users_geneloneri'),

]



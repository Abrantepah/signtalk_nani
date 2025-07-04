from django.urls import path

from . import views

urlpatterns = [
    path('home', views.index, name='home'),
    path('select_user', views.select_user, name='select_user'),
    path('select_solution', views.select_solution, name='select_solution'),
    path('learn_main', views.learn_main, name='learn_main'),
    path('learn_page/<str:pk>/', views.learn_page, name='learn_page'),
    path('hospital_page', views.hospital_page, name='hospital_page'),
    path('community', views.community, name='community'),
    path('community_chat', views.community_chat, name='community_chat'),
    path("audio_to_sign", views.audioToSign, name="audioToSign"),
    path('textsign', views.textsign, name='textsign'),
    path('signtext', views.signtext, name='signtext'),
    path('start_recording', views.start_recording, name='start_recording'),
    path('stop_recording', views.stop_recording, name='stop_recording'),
    path('learn', views.learn, name='learn'),
    path('practice', views.practice, name='practice'),
    path('live_feed', views.livefeed, name="live_feed"),
    path('update_session', views.update_session, name="update_session"),
]


from django.conf.urls import url,include
from . import views

urlpatterns= [
	url(r'^website/$',views.website,name='website'),
	url(r'^schedule/$',views.schedule,name='schedule'),
]
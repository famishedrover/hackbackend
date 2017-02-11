from django.shortcuts import render
from django.http import HttpResponse
import json
import requests
from django.views.decorators.csrf import csrf_exempt
from . import scraping_api
# from . import ml
from ml import *

@csrf_exempt
def website(request):
	print "Request Begins"
	print request.META
	print ".........."

	response={}
	if request.method=='POST':
		try:
			obj={}
			for i in request.POST:
				print "naice"
				print i
				obj[str(i)]=str(request.POST[i])
			print "Object is "
			print obj
			score=scraping_api.scorer(json.loads(obj['PostData']))
			print "Score obtained is "+score
			f=open('web_score.txt','w')   		#Option to keep the score in a Databse #Currently keeping in a file
			f.write(str(score))
			f.close()


			# f=open('web_score.txt','r')
			# print("score from file is "+str(f.read()))
			# f.close()
			response={'Status':'Successfully got the Score'}
		except:
			response={'Status':'An unknown error '}

	else:
		response={'Status':'Error please send a POST Request'}
	
	print "Done"
	return HttpResponse(json.dumps(response))

@csrf_exempt
def schedule(request):
	response={'status':1}
	req_obj={}
	#
	#{"sscore": str(mylist[0]),
        #"sleep_hours": str(mylist[1]),
       # "work_hours": str(mylist[2]),
       # "freetime": str(mylist[3]),
       # "holperyear": str(mylist[4])
       # }
	print "Here is the Request"
	if request.method=='POST':
		try:
			print "The POST data is"
			
			req_obj=json.loads(request.POST['PostData'])
			
			# for i in json.loads(request.POST['PostData']):
			# 	req_obj[str(i)]=str(request.POST[i])
			f=open('web_score.txt','r')
			score=str(f.read())
			f.close()
			req_obj['sscore']=int(score)
			for i in req_obj:
				req_obj[i]=int(req_obj[i])

			print "Check"
			print req_obj

			optimal={}
			
			optimal=predict_optimal_answer(req_obj)
			print optimal
			
			for i in optimal:
				response[i]=optimal[i]

			print response
			#send req_object to Mudit's Function
			#Get the response from Mudit's Function
			#Now put the key :value pairs in response
			#return this response (JSON) HttpResponse

		except:
			response['status']=0;
			pass
	else:
		response['status']=0
	return HttpResponse(json.dumps(response))




	

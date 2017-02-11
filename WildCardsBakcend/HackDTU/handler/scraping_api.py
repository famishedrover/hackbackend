import os 
import sys 
import requests
from bs4 import BeautifulSoup 
import json

#site /robot.txt

obj={}



URL="https://depression.org.nz/is-it-depression-anxiety/self-test/anxiety-test/result"

#incoming json of the form {'1':'0','2':'1','3':'3'}

def scorer(incoming_obj):	
	print "Scraping Reached"
	print incoming_obj

	for j in incoming_obj:
		print str(j)+str(incoming_obj[j])
		obj['questions['+j+']']=incoming_obj[j]
		
	#print obj
	# try:
	re=requests.get(url=URL,params=obj)
	#print "Url"
	#print re.url
	soup=BeautifulSoup(re.text,'html.parser')
	#print soup.find_all(class_='test__result')[0].find('span',class_='number')
	numbers=soup.find_all(class_='test__result')[0]
	#print numbers
	numbers=numbers.find('span',class_='number').string
	#print numbers.split('Your score is')
	substring=numbers.split('Your score is ')[1]
	ans=substring[0:2]
	print "Score is "+ str(ans)
	return str(ans)
	# except:
	# 	return str(0)
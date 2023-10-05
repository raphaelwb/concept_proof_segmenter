from django.shortcuts import render
import openai
import json
import re
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import traceback
import ssl
from mysite.settings import KEY, ENABLED, KEYUSER
openai.api_key = KEY
enableGTP = False
if (ENABLED == "TRUE"):
    enableGTP = True
from bs4 import BeautifulSoup
import requests
import topic_segmenter
import time
from django.http import HttpResponseForbidden

ssl._create_default_https_context = ssl._create_unverified_context

def loadData(url):
    html = requests.get(url,verify=False).content
    htmlText = requests.get(url,verify=False).text
    
    with open('content.html', 'wb+') as f:
        f.write(html)

    soup = BeautifulSoup(html)
    text = soup.get_text()
    #print(text)

    data = topic_segmenter.segment_text(text,1)
    #print("Size:",len(data))
    print("<<<<<-------------------------------------------------------------->>>>>>>")
    anchor = []
    htmlData = []
    jsonData = []

    for d in data:
        try:
            d = re.sub("\s\s+" , " ",d)
            d = d.replace("\n",".")
            response = ""
            #print(d)
            #print() 
            if ( enableGTP ):
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Crie um cabeçalho de texto puro com esse texto:"+d,
                    temperature=0.5,
                    max_tokens=60,
                    top_p=1.0,
                    frequency_penalty=0.8,
                    presence_penalty=0.0,
                ) 
                response = response.choices[0].text
            else:
                response = d[0:50]
            
            if ( "#" in response ):
                response = response.split("#")
                response = response[1]
            response = response.replace("\n\n<h1>","")
            response = response.replace(" \n",".")
            response = response.replace("</h1>","")
            #response = response.strip()
            if ( "?" in response ):
                response = response[:response.find("?")]+"?"
            elif ( "!" in response ):
                response = response[:response.find("!")]+"!"
            elif ( "." in response ):
                response = response[:response.find(".")]+"."
            elif ( "-" in response ):
                response = response[:response.find("-")]+"."
            elif ( ";" in response ):
                response = response[:response.find(";")]+"."
            elif ( ":" in response ):
                response = response[:response.find(":")]+"."   
        
            anchor.append({"title":response})
            htmlData.append({"title":response,"text":d})

            anc = d[0:20].strip()
            if ( htmlText.find(anc) != -1 ):
                jsonData.append({"title":response,"text":anc})
            else:
                #Gambi pra corrigir um problema no processo, pois o segment_text recebe o TEXTO sem TAGS
                #Ele junta palavras de mais de uma TAG do HTML, tornando impossível achar a mesma posteriormente
                print("NOT FOUND:",anc)
                words = anc.split()
                anc = ""
                if ( len(words) > 3 ):
                    for w in range(0,len(words)//2):
                        anc += words[w] + " "
                    anc = anc.strip()    
                    print("New ANC:["+anc+"]")
                    if ( htmlText.find(anc) != -1 ):
                        jsonData.append({"title":response,"text":anc})
                        print("CORRECTED")
                    else:
                        print("NOT FOUND AFTER GAMBI:",anc)


        except Exception as e: 
            traceback.print_exc()

    #print("-----------------------------------------------------------")
    #print(jsonData)
    print("FINISHED")
    return jsonData,htmlData,anchor

# Create your views here.
@csrf_exempt
def apiView(request):
    print(request.POST.items())
    url = None
    keyUser = None
    for a in request.POST.items():
        print(a[0])
        params = json.loads(a[0])
        url = params["url"]
        keyUser = params["keyuser"]
    print(keyUser)

    if ( keyUser == KEYUSER ):
        if ( url != None ):
            try:
                
                jsonData,htmlData,anchor = loadData(url)
                #print(jsonData)
                #print("OK")
                return JsonResponse(jsonData,safe=False)
            except Exception as e:
                traceback.print_exc()
                print("NOT OK")
                return JsonResponse([{"error":"process"}],safe=False)
    print("PARAM NOT OK")
    return JsonResponse([{"error":"parameter invalid"}],safe=False)

def indexView(request):
    if ( "url" in request.POST):
        print(request.POST["keyuser"])
        print(KEYUSER)
        if ( "keyuser" in request.POST and request.POST["keyuser"] == KEYUSER ):
            try:
                jsonData,htmlData,anchor = loadData(request.POST["url"])
                return render(request, 'data.html',{"data":htmlData,"anchor":anchor})
            except Exception as e:
                traceback.print_exc()
                return render(request, 'error.html',{"msg":str(e)})
        else:
            return HttpResponseForbidden()
    return render(request, 'index.html')
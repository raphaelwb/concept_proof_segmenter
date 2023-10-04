import os
import openai
import json
import re

openai.api_key = "******"
enableGTP = False

from bs4 import BeautifulSoup
import requests
import topic_segmenter
html = requests.get("https://noticias.r7.com/internacional/uma-semana-depois-terremoto-na-turquia-e-siria-ja-matou-mais-de-35-mil-pessoas-13022023").content
soup = BeautifulSoup(html)
print("-----------------------------------------------")
text = soup.get_text()
#print(text)
print("-----------------------------------------------")
data = topic_segmenter.segment_text(text,1)
print("Size:",len(data))
print("<<<<<-------------------------------------------------------------->>>>>>>")
for d in data:
    d = re.sub("\s\s+" , " ",d)
    d = d.replace("\n",".")
    response = ""
    print(d)
    print() 
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
      print(response)
      response = response.choices[0].text
    else:
      response = "\n\n#Entenda os Efeitos de um El Ni\u00f1o Cl\u00e1ssico no Brasil\n\nDepois do La Ni\u00f1a, ser\u00e1 poss\u00edvel um El Ni\u00f1o? O Clima que Queremos descobre! Al\u00e9m disso, muitas em"
    
    response = response.replace("\n\n#","")
    response = response.replace("\n\n<h1>","")
    response = response.replace(" \n",".")
    response = response.replace("\n",".")
    response = response.replace("</h1>","")
    response = response.strip()
    if ( "?" in response ):
      response = response[:response.find("?")]+"?"
    elif ( "!" in response ):
      response = response[:response.find("!")]+"!"
    elif ( "." in response ):
      response = response[:response.find(".")]+"."
    elif ( "-" in response ):
      response = response[:response.find("-")]+"."
   
    print("-->"+response)
    print("--------------------------------------------------------------------------------------------------------------------------------")
    print(d)
    print() 
    print()

    


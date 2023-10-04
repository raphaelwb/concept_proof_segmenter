from bs4 import BeautifulSoup
import requests
import topic_segmenter
html = requests.get("https://www.climatempo.com.br/").content
soup = BeautifulSoup(html)
print("-----------------------------------------------")
text = soup.get_text()
#print(text)
print("-----------------------------------------------")
data = topic_segmenter.segment_text(text,1)
print("Size:",len(data))
for d in data:
    print("----------------")
    print(d)
# Topic Segmenter

## Prepare the application

```shell script
python3 -m venv venv
pip3 install -r requirements.txt
```

## Running the application in dev mode

```shell script
python3 manage.py runserver
```

## Running the server mode

```shell script
ngrok http 8000
python3 manage.py runserver
```

Ngrok should run in a different Terminal   
Get ngrok Address   
Change in mysite/settings.py (ALLOWED_HOSTS and CSRF_TRUSTED_ORIGINS)   
Change in chromeExtension/content.js (api=?)   

## Install Chrome Plugin

Extensions >> Developer Mode ON   
Extesions >> Load Unpacked >> Select    directory (chromeExtension)   
Use Alt+Shift+1 or click the Icon   

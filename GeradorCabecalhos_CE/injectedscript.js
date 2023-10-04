console.log("Receiver content");

api = "http://127.0.0.1:8000";

function playSound() {
    let url = chrome.runtime.getURL('audio.html');

    // set this string dynamically in your code, this is just an example
    // this will play success.wav at half the volume and close the popup after a second
    url += '?volume=0.5&src=success.wav&length=1000';

    chrome.windows.create({
        type: 'popup',
        focused: true,
        top: 1,
        left: 1,
        height: 1,
        width: 1,
        url,
    })

}

(async () => {
    const response = await fetch(api+"/api", {
    mode: 'no-cors',
    method: 'POST',
    body: JSON.stringify({
        url: window.location.toString(),
        keyuser: "*****"
    }),
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
    },
    })

    const data = await response.json()

    //console.log(data);

    var links = ""

    var c = 1
    data.forEach(function (entry) {
    console.log(entry.text);
    newText = '<mark id="markpos_'+c+'">[' + entry.title + ']</mark>' + entry.text;
    var found = document.body.innerHTML.indexOf(entry.text);
    console.log("Found ["+entry.text+"] at"+found);
    
    document.body.innerHTML = document.body.innerHTML.replace(entry.text, newText);
    links += '<a href="#markpos_'+c+'">'+entry.title+'</a><BR>';
    c+=1;
    });

    document.body.innerHTML = "<B>Links para seções da página</B><BR>"+links + document.body.innerHTML;
    console.log(links);

    window.open("https://media.geeksforgeeks.org/wp-content/uploads/20190531135120/beep.mp3", "_blank")

})()
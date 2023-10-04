// A2Z F17
// Daniel Shiffman
// http://shiffman.net/a2z
// https://github.com/shiffman/A2Z-F17


chrome.action.onClicked.addListener(() => {
  chrome.tabs.query({active: true, currentWindow: true}).then(([tab]) => {
    chrome.scripting.executeScript({
      target: {tabId: tab.id},
      files: ['injectedscript.js'],
    });
  });
});
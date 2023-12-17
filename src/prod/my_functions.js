var tag = document.createElement('script');
tag.src = 'https://www.youtube.com/iframe_api';
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
var player, seconds = 0;


function onPlayerReady(event) {
    event.target.playVideo();
}
function seek(sec){
    var documentContainer = document;
    var iframe            = documentContainer.getElementById('player');
    console.log("iframe");
    console.log(iframe);
    player = new YT.Player(iframe, {
        events: {
          'onReady': onPlayerReady
        }
    });
    if(player){
      player.seekTo(sec, true);
    }
}

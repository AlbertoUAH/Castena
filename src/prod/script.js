var tag = document.createElement('script');
tag.src = 'https://www.youtube.com/iframe_api';
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

function onYouTubeIframeAPIReady() {
    var iframe = document.getElementById('player');
    player = new YT.Player(iframe, {
        events: {
          'onReady': onPlayerReady
        }
      });
}
function onPlayerReady(event) {
    event.target.playVideo();
}
var iframe = document.getElementById('player');
player = new YT.Player(iframe, {
        events: {
          'onReady': onPlayerReady
        }
      });
function seek(sec){
    if(player){
        player.seekTo(sec, true);
    }
}

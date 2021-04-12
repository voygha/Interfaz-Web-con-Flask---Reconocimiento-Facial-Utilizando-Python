navigator.mediaDevices.getUserMedia({ audio: false, video: true}).then((stream)=>{
    console.log(stream)

    let video = document.getElementById('video')
    video.src

 

}).catch((err)=>console.log(err))
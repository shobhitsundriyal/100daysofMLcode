let mobilenet;//Pretrained model is mobile net

function modelReady(){
    console.log('Model is Ready')
    mobilenet.predict(myimage, gotResults);
}

function gotResults(error,result){
    if(error){
        console.error(error)
    }
    else{
        console.log(result)
        let label = result[0].className;
        let probability = result[0].probability;
        fill(0);
        textSize(64);
        text(label,0,height-100)
        
        createP(label)
        createP(probability)
    }
}

function imageReady(){
    image(myimage, 0, 0, width, height);
}

function setup() {
    createCanvas(640, 480);
    myimage=createImg('images/what.jpg', imageReady)
    myimage.hide()
    background(0);
    //image(myimage, 0, 0)
    mobilenet = ml5.imageClassifier('MobileNet', modelReady)
  
  }
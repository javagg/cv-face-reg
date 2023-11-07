<script lang="ts">
  import cv from "@techstark/opencv-js";
  import { onMount } from "svelte";
  import {createFileFromUrl} from './lib/utl'

  let proto = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy_lowres.prototxt';
  let weights = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel';
  let recognModel = 'https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7';
  let videoSource = null;
  let output = null;
  let loading = false;
  let cap = null;
  let frame = null;
  let frameBGR = null;
  let message = ""

  let people = [];

  let modelLoaded = false
  let cameraOpened = false

  var isRunning = false;

  async function openCamera() {
    try {
      loading = true;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      videoSource.srcObject = stream;
      videoSource.play();
      loading = false;
      cameraOpened = true

      cap = new cv.VideoCapture(videoSource);
     frame = new cv.Mat(videoSource.height, videoSource.width, cv.CV_8UC4);
     frameBGR = new cv.Mat(videoSource.height, videoSource.width, cv.CV_8UC3);
     
    } catch (error) {
      console.log(error);
    }
  }
 
  const obtenerVideoCamara = async () => {
    // try {
    //   // loading = true;
    //   // const stream = await navigator.mediaDevices.getUserMedia({
    //   //   video: true,
    //   // });
    //   // videoSource.srcObject = stream;
    //   // videoSource.play();
    //   // loading = false;
    //   // cameraOpened = true

    //  cap = new cv.VideoCapture(videoSource);
    //  frame = new cv.Mat(videoSource.height, videoSource.width, cv.CV_8UC4);
    //  frameBGR = new cv.Mat(videoSource.height, videoSource.width, cv.CV_8UC3);
    // //  await run()
    // } catch (error) {
    //   console.log(error);
    // }
  };

  function add() {
		people = people.concat({
			done: false,
			text: ''
		});
	}

  const FPS = 30;  // Target number of frames processed per second.
  function captureFrame() {
    console.log("aaaaa")
      var begin = Date.now();
      cap.read(frame);
      cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);
  
      var faces = detectFaces(frameBGR);
      faces.forEach(function(rect) {
        cv.rectangle(frame, {x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y + rect.height}, [0, 255, 0, 255]);
  
        var face = frameBGR.roi(rect);
        var name = recognize(face);
        cv.putText(frame, name, {x: rect.x, y: rect.y}, cv.FONT_HERSHEY_SIMPLEX, 1.0, [0, 255, 0, 255]);
      });
  
      cv.imshow(output, frame);
  
      // Loop this function.
      if (isRunning) {
        var delay = 1000 / FPS - (Date.now() - begin);
        setTimeout(captureFrame, delay);
      }
    };

  onMount(async ()=> {
    await loadModels()
    await openCamera()
    await run()
  })
	// function clear() {
	// 	people = people.filter((t) => !t.done);
	// }

  const addPersion = async () => {
    const rects = detectFaces(frameBGR);
    if (rects.length > 0) {
      let face = frameBGR.roi(rects[0]);
      let name = prompt('Say your name:');

      people = people.concat({
        done: false,
        text: ''
      });

      //   var cell = document.getElementById("targetNames").insertCell(0);
      //   cell.innerHTML = name;
  
      //   persons[name] = face2vec(face).clone();
  
      //   var canvas = document.createElement("canvas");
      //   canvas.setAttribute("width", 96);
      //   canvas.setAttribute("height", 96);
      //   var cell = document.getElementById("targetImgs").insertCell(0);
      //   cell.appendChild(canvas);
  
      //   var faceResized = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC3);
      //   cv.resize(face, faceResized, {width: canvas.width, height: canvas.height});
      //   cv.cvtColor(faceResized, faceResized, cv.COLOR_BGR2RGB);
      //   cv.imshow(canvas, faceResized);
      //   faceResized.delete();
    }
  }
  // let buildInfo: string = "";

  var netDet = undefined, netRecogn = undefined;
  var persons = {};
  
  function detectFaces(img) {
    var blob = cv.blobFromImage(img, 1, {width: 192, height: 144}, [104, 117, 123, 0], false, false);
    netDet.setInput(blob);
    var out = netDet.forward();
  
    var faces = [];
    for (var i = 0, n = out.data32F.length; i < n; i += 7) {
      var confidence = out.data32F[i + 2];
      var left = out.data32F[i + 3] * img.cols;
      var top = out.data32F[i + 4] * img.rows;
      var right = out.data32F[i + 5] * img.cols;
      var bottom = out.data32F[i + 6] * img.rows;
      left = Math.min(Math.max(0, left), img.cols - 1);
      right = Math.min(Math.max(0, right), img.cols - 1);
      bottom = Math.min(Math.max(0, bottom), img.rows - 1);
      top = Math.min(Math.max(0, top), img.rows - 1);
  
      if (confidence > 0.5 && left < right && top < bottom) {
        faces.push({x: left, y: top, width: right - left, height: bottom - top})
      }
    }
    blob.delete();
    out.delete();
    return faces;
  };
  
  function face2vec(face) {
    var blob = cv.blobFromImage(face, 1.0 / 255, {width: 96, height: 96}, [0, 0, 0, 0], true, false)
    netRecogn.setInput(blob);
    var vec = netRecogn.forward();
    blob.delete();
    return vec;
  };
  
  function recognize(face) {
    const vec = face2vec(face);
    var bestMatchName = 'unknown';
    var bestMatchScore = 0.5;  // Actually, the minimum is -1 but we use it as a threshold.
    for (name in persons) {
      var personVec = persons[name];
      var score = vec.dot(personVec);
      if (score > bestMatchScore) {
        bestMatchScore = score;
        bestMatchName = name;
      }
    }
    vec.delete();
    return bestMatchName;
  };
  
  async function loadModels() {
    message = "Downloading face_detector.prototxt"
    await createFileFromUrl('face_detector.prototxt', proto)
    message = 'Downloading face_detector.caffemodel'
    await  createFileFromUrl('face_detector.caffemodel', weights)
    message = 'Downloading OpenFace model'
    await createFileFromUrl('face_recognition.t7', recognModel)
    message = ""

    modelLoaded = true
    netDet = cv.readNetFromCaffe('face_detector.prototxt', 'face_detector.caffemodel');
    netRecogn = cv.readNetFromTorch('face_recognition.t7');
  };
  
  async function run() {
    isRunning = true
    captureFrame()
  }
  // function main() {
  //   // var output = document.getElementById('output');
  //   // var camera = document.createElement("video");
  //   // camera.setAttribute("width", output.width);
  //   // camera.setAttribute("height", output.height);
  
  //   // Get a permission from user to use a camera.
  //   // navigator.mediaDevices.getUserMedia({video: true, audio: false})
  //   //   .then(function(stream) {
  //   //     camera.srcObject = stream;
  //   //     camera.onloadedmetadata = function(e) {
  //   //       camera.play();
  //   //     };
  //   // });
  
  //   // //! [Open a camera stream]
  //   // var cap = new cv.VideoCapture(camera);
  //   // var frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
  //   // var frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
  //   // //! [Open a camera stream]
  
  //   //! [Add a person]
  //   // document.getElementById('addPersonButton').onclick = function() {
  //   //   var rects = detectFaces(frameBGR);
  //   //   if (rects.length > 0) {
  //   //     var face = frameBGR.roi(rects[0]);
  
  //   //     var name = prompt('Say your name:');
  //   //     var cell = document.getElementById("targetNames").insertCell(0);
  //   //     cell.innerHTML = name;
  
  //   //     persons[name] = face2vec(face).clone();
  
  //   //     var canvas = document.createElement("canvas");
  //   //     canvas.setAttribute("width", 96);
  //   //     canvas.setAttribute("height", 96);
  //   //     var cell = document.getElementById("targetImgs").insertCell(0);
  //   //     cell.appendChild(canvas);
  
  //   //     var faceResized = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC3);
  //   //     cv.resize(face, faceResized, {width: canvas.width, height: canvas.height});
  //   //     cv.cvtColor(faceResized, faceResized, cv.COLOR_BGR2RGB);
  //   //     cv.imshow(canvas, faceResized);
  //   //     faceResized.delete();
  //   //   }
  //   // };
  //   //! [Add a person]
  
  //   // var isRunning = false;

   
  //   // document.getElementById('startStopButton').onclick = function toggle() {
  //   //   if (isRunning) {
  //   //     isRunning = false;
  //   //     document.getElementById('startStopButton').innerHTML = 'Start';
  //   //     document.getElementById('addPersonButton').disabled = true;
  //   //   } else {
  //   //     function run() {
  //   //       isRunning = true;
  //   //       captureFrame();
  //   //       document.getElementById('startStopButton').innerHTML = 'Stop';
  //   //       document.getElementById('startStopButton').disabled = false;
  //   //       document.getElementById('addPersonButton').disabled = false;
  //   //     }
  //   //     if (netDet == undefined || netRecogn == undefined) {
  //   //       document.getElementById('startStopButton').disabled = true;
  //   //       loadModels(run);  // Load models and run a pipeline;
  //   //     } else {
  //   //       run();
  //   //     }
  //   //   }
  //   // };
  
  //   // document.getElementById('startStopButton').disabled = false;
  // };


</script>
<!-- svelte-ignore a11y-media-has-caption -->
<!-- <video  class="width: 400px; height: 300px" bind:this={videoSource} /> -->
<button on:click={obtenerVideoCamara} disabled='{!modelLoaded}' >Start</button>
<button on:click={addPersion} disabled={!modelLoaded}>Add</button>

<ul class="todos">
  {#each people as person}
    <li class:done={person.done}>
      <!-- <input
        type="checkbox"
        checked={todo.done}
      />

      <input
        type="text"
        placeholder="What needs to be done?"
        value={todo.text}
      /> -->
    </li>
  {/each}
</ul>
<div>
  {#if loading}
    <h1>LOADING</h1>
  {/if}
  <!-- svelte-ignore a11y-media-has-caption -->
  <video class="width: 400px; height: 300px" bind:this={videoSource} />
  <button on:click={obtenerVideoCamara}>CLICK</button>
  <p>Status: {message}</p>

  <canvas  class="width: 400px; height: 300px" bind:this={output} />
</div>

<style>
  /* .build-info {
    width: 600px;
    height: 400px;
  } */
</style>

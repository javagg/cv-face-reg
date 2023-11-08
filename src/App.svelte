<script lang="ts">
  import cv from "@techstark/opencv-js";
  import type { Net } from "@techstark/opencv-js";

  import { onDestroy, onMount } from "svelte";
  import { createFileFromUrl } from "./lib/util";

  let proto = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy_lowres.prototxt";
  let weights = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel";
  let recognModel = "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7";
  let videoSource = null;
  let output = null;
  let loading = false;
  let cap: cv.VideoCapture = null;
  let frame: cv.Mat = null;
  let frameBGR: cv.Mat = null;
  let message = "";

  let people = [];

  let modelLoaded = false;

  let timer: number = null;
  var isRunning = false;


  async function openCamera(): Promise<HTMLVideoElement> {
    const camera: HTMLVideoElement = document.createElement("video");
    camera.setAttribute("width", output.width);
    camera.setAttribute("height", output.height);

    try {
      loading = true;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      camera.srcObject = stream;
      camera.play();
      loading = false;

      cap = new cv.VideoCapture(camera);
      frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
      frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
      return camera
    } catch (error) {
      console.log(error);
      return null;
    }
  }

  function add() {
    people = people.concat({
      done: false,
      text: "",
    });
  }

  const FPS = 30; // Target number of frames processed per second.
  function captureFrame() {
    const begin = Date.now();
    cap.read(frame);
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    const faces = detectFaces(frameBGR);
    faces.forEach(function (rect) {
      cv.rectangle(
        frame,
        { x: rect.x, y: rect.y },
        { x: rect.x + rect.width, y: rect.y + rect.height },
        [0, 255, 0, 255]
      );

      const face = frameBGR.roi(rect);
      const name = recognize(face);
      cv.putText(
        frame,
        name,
        { x: rect.x, y: rect.y },
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        [0, 255, 0, 255]
      );
    });

    cv.imshow(output, frame);
  }

  onMount(async () => {
    await loadModels();
    videoSource = await openCamera();
    timer = setInterval(async()=>{
      await run();
    }, 1000)
  });

  onDestroy(()=>{
    clearInterval(timer)
  })
  // function clear() {
  // 	people = people.filter((t) => !t.done);
  // }

  const addPersion = async () => {
    const rects = detectFaces(frameBGR);
    if (rects.length > 0) {
      let face = frameBGR.roi(rects[0]);
      let name = prompt("Say your name:");

      people = people.concat({
        done: false,
        text: "",
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
  };

  let netDet: Net = undefined;
  let netRecogn: Net = undefined;
  var persons = {};

  function detectFaces(img) {
    var blob = cv.blobFromImage(
      img,
      1,
      { width: 192, height: 144 },
      [104, 117, 123, 0],
      false,
      false
    );
    netDet.setInput(blob);
    var out = netDet.forward();

    const faces = [];
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
        faces.push({
          x: left,
          y: top,
          width: right - left,
          height: bottom - top,
        });
      }
    }
    blob.delete();
    out.delete();
    return faces;
  }

  function face2vec(face) {
    var blob = cv.blobFromImage(
      face,
      1.0 / 255,
      { width: 96, height: 96 },
      [0, 0, 0, 0],
      true,
      false
    );
    netRecogn.setInput(blob);
    var vec = netRecogn.forward();
    blob.delete();
    return vec;
  }

  function recognize(face) {
    const vec = face2vec(face);
    var bestMatchName = "unknown";
    var bestMatchScore = 0.5; // Actually, the minimum is -1 but we use it as a threshold.
    for (const person in persons) {
      var personVec = persons[person];
      var score = vec.dot(personVec);
      if (score > bestMatchScore) {
        bestMatchScore = score;
        bestMatchName = person;
      }
    }
    vec.delete();
    return bestMatchName;
  }

  async function loadModels() {
    message = "Downloading face_detector.prototxt";
    await createFileFromUrl("face_detector.prototxt", proto);
    message = "Downloading face_detector.caffemodel";
    await createFileFromUrl("face_detector.caffemodel", weights);
    message = "Downloading OpenFace model";
    await createFileFromUrl("face_recognition.t7", recognModel);
    message = "";

    modelLoaded = true;
    netDet = cv.readNetFromCaffe(
      "face_detector.prototxt",
      "face_detector.caffemodel"
    );
    netRecogn = cv.readNetFromTorch("face_recognition.t7");
  }

  async function run() {
    isRunning = true;
    captureFrame();
  }

</script>

<main>
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
    <p>Status: {message}</p>
    <canvas class="can" bind:this={output} />
  </div>
</main>

<style>
  .can {
    width: 600px;
    height: 400px;
  } 
</style>


import cv from "@techstark/opencv-js";

export async function createFileFromUrl(path:string, url:string,) {
    const response = await fetch(url)
    const data = await response.arrayBuffer(); 
    cv.FS_createDataFile('/', path, new Uint8Array(data), true, false, false);
}
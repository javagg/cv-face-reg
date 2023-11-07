
import cv from "@techstark/opencv-js";

export async function createFileFromUrl(path:string, url:string,) {
    const result = await fetch(url)
    const data = result.blob
    cv.FS_createDataFile('/', path, data, true, false, false);
}
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  isImageFile,
  isSupportedMediaFile,
  isVideoFile,
  summarizeMediaUpload,
} from "../../src/lib/reconstruct/upload-files.ts";

describe("upload-files media classification", () => {
  it("recognizes image files by mime type or extension", () => {
    assert.equal(isImageFile({ type: "image/jpeg", name: "photo.jpg" }), true);
    assert.equal(isImageFile({ type: "", name: "scan.HEIC" }), true);
    assert.equal(isImageFile({ type: "application/pdf", name: "doc.pdf" }), false);
  });

  it("recognizes video files by mime type or extension", () => {
    assert.equal(isVideoFile({ type: "video/mp4", name: "walkthrough.mp4" }), true);
    assert.equal(isVideoFile({ type: "", name: "capture.MOV" }), true);
    assert.equal(isVideoFile({ type: "video/webm", name: "clip.webm" }), true);
    assert.equal(isVideoFile({ type: "image/png", name: "frame.png" }), false);
  });

  it("accepts both images and videos as supported media and rejects non-media", () => {
    assert.equal(isSupportedMediaFile({ type: "image/webp", name: "a.webp" }), true);
    assert.equal(isSupportedMediaFile({ type: "video/mp4", name: "b.mp4" }), true);
    assert.equal(isSupportedMediaFile({ type: "text/plain", name: "notes.txt" }), false);
  });

  it("summarizes photo-only, video-only, and mixed uploads accurately", () => {
    const photos = summarizeMediaUpload([
      { type: "image/jpeg", name: "1.jpg" },
      { type: "image/jpeg", name: "2.jpg" },
    ]);
    assert.equal(photos.imageCount, 2);
    assert.equal(photos.videoCount, 0);
    assert.match(photos.busyLabel, /2 photos/);

    const videos = summarizeMediaUpload([{ type: "video/mp4", name: "room.mp4" }]);
    assert.equal(videos.imageCount, 0);
    assert.equal(videos.videoCount, 1);
    assert.match(videos.busyLabel, /1 video.*keyframes/);

    const mixed = summarizeMediaUpload([
      { type: "image/png", name: "1.png" },
      { type: "video/quicktime", name: "2.mov" },
    ]);
    assert.equal(mixed.acceptedCount, 2);
    assert.match(mixed.busyLabel, /1 photo.*1 video/);
  });
});

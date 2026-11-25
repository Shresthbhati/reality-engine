/**
 * Pure media file classification for reconstruction uploads.
 * Kept React-free so contract tests can verify image + video acceptance directly.
 */

const IMAGE_EXT = /\.(jpe?g|png|webp|heic|heif|tiff?|bmp)$/i;
const VIDEO_EXT = /\.(mp4|mov|m4v|webm|mkv|avi)$/i;

export function isImageFile(file: Pick<File, "type" | "name">): boolean {
  return file.type.startsWith("image/") || IMAGE_EXT.test(file.name);
}

export function isVideoFile(file: Pick<File, "type" | "name">): boolean {
  return file.type.startsWith("video/") || VIDEO_EXT.test(file.name);
}

export function isSupportedMediaFile(file: Pick<File, "type" | "name">): boolean {
  return isImageFile(file) || isVideoFile(file);
}

export function summarizeMediaUpload(files: ReadonlyArray<Pick<File, "type" | "name">>): {
  acceptedCount: number;
  imageCount: number;
  videoCount: number;
  busyLabel: string;
} {
  let imageCount = 0;
  let videoCount = 0;
  for (const f of files) {
    if (isVideoFile(f)) videoCount += 1;
    else if (isImageFile(f)) imageCount += 1;
  }
  const acceptedCount = imageCount + videoCount;
  if (videoCount > 0 && imageCount === 0) {
    return {
      acceptedCount,
      imageCount,
      videoCount,
      busyLabel: `Uploading ${videoCount} video${videoCount === 1 ? "" : "s"} & extracting keyframes…`,
    };
  }
  if (videoCount > 0 && imageCount > 0) {
    return {
      acceptedCount,
      imageCount,
      videoCount,
      busyLabel: `Uploading ${imageCount} photo${imageCount === 1 ? "" : "s"} & ${videoCount} video${videoCount === 1 ? "" : "s"}…`,
    };
  }
  return {
    acceptedCount,
    imageCount,
    videoCount,
    busyLabel: `Uploading ${imageCount} photo${imageCount === 1 ? "" : "s"}…`,
  };
}

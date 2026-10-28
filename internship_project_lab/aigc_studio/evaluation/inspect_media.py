"""Decode complete outputs and write diagnostic contact sheets; no semantic scoring."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def inspect(path, output):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
        with Image.open(path) as image:
            image.load()
            image = image.convert('RGB')
            pixels = np.asarray(image, dtype=np.float32)
            result = {'kind': 'image', 'width': image.width, 'height': image.height, 'decoded': True,
                      'pixel_std': float(pixels.std()), 'mean_brightness': float(pixels.mean()),
                      'near_black_fraction': float((pixels.max(axis=2) < 3).mean())}
            image.thumbnail((512, 512))
            image.save(output / 'preview.jpg')
            result['preview'] = str(output / 'preview.jpg')
            return result
    thumbnails, changes, previous, times = [], [], None, []
    count, black, width, height = 0, 0, 0, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(video=0):
            image = frame.to_image().convert('RGB')
            width, height = image.size
            small = np.asarray(image.resize((128, 128)), dtype=np.float32)
            black += int(small.max() < 3)
            if previous is not None:
                changes.append(float(np.abs(small - previous).mean() / 255))
            previous = small
            thumb = image.copy()
            thumb.thumbnail((256, 160))
            thumbnails.append(thumb)
            times.append(float(frame.time) if frame.time is not None else count / (fps or 1))
            count += 1
    if not count:
        raise ValueError('Video has no decodable frames')
    indices = np.linspace(0, count - 1, min(8, count)).round().astype(int)
    sheet = Image.new('RGB', (4 * 256, 2 * 190), 'white')
    draw = ImageDraw.Draw(sheet)
    for pos, idx in enumerate(indices):
        x, y = pos % 4 * 256, pos // 4 * 190
        sheet.paste(thumbnails[idx], (x, y))
        draw.text((x + 6, y + 164), f'frame {idx}, {times[idx]:.2f}s', fill='black')
    sheet.save(output / 'contact.jpg')
    return {'kind': 'video', 'decoded': True, 'width': width, 'height': height, 'frames': count, 'fps': fps,
            'duration_seconds': count / fps if fps else None, 'black_frames': black,
            'mean_frame_pixel_change': float(np.mean(changes)) if changes else 0,
            'preview': str(output / 'contact.jpg'), 'diagnostic_note': 'Pixel change is not a motion-quality or semantic score.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('output')
    args = parser.parse_args()
    print(json.dumps(inspect(args.path, args.output)))

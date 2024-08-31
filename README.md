# ImageDuplicateChecker

A simple Python application with a GUI (PyQt5) for finding and managing duplicate images in a specified folder I've adapted it from a script that I use to deduplicate image datasets.

This program uses [perceptual hashing](https://en.wikipedia.org/wiki/Perceptual_hashing) (including dHash) to compare and match images that are similar but not identical (e.g., grayscaled or resized images).

## Features

- Select a folder to scan for duplicate images.
- Support for various image formats including JPEG, PNG, GIF, BMP, TIFF, WebP, ICO, PPM, TGA
- Support for common RAW image formats (RAW, ARW, CR2, NEF, ORF, RW2, DNG)
- Support basic transformations (flip horizontally/vertically, rotation 90°/180°/270°) 
- Adjust hash size for controlling sensitivity of duplicate detection
- Display duplicate images in groups
- Remove selected duplicate images by moving them to the trash (supports OS X, Windows and Linux).
- Pagination for easier navigation of results
- Progress bar to show scanning progress
- Multithreading (configurable threads) + LRU caching of image hashes for improved performance
- Batching for lower memory footprint
- Option to check subfolders
- Preferences saving and loading

## TODO

- Implement database backend to enhance performance for large image collections
- ~~Add configurable settings for threads, batches, and cache~~ [Done]
- ~~Deleted images are moved to the trash instead of being directly erased on all OSes.~~ [Done]
- ~~Expand support for additional image formats~~ [Done]
- ~~Option to only check certain image extensions~~ [Done]
- ~~Improve comparison to handle basic image transformations (e.g., rotated images)~~ [Done]

## Requirements

- Python 3.6+
- PyQt5
- Pillow
- ImageHash
- Rawpy
- Numpy
- opencv-python

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-duplicate-checker.git
   ```

2. Navigate to the project directory:
   ```
   cd image-duplicate-checker
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:

```
python image_duplicate_checker.py
```

1. Click "Select Folder" to choose the directory you want to scan.
2. Adjust the hash size if needed (lower values increase sensitivity but may produce more false positives).
3. Click "Check Duplicates" to start the scanning process.
4. Once complete, review the duplicate groups displayed.
5. Select images you want to remove and click "Remove Selected" to delete them (there is a confirmation box if you want to erase > 1 image). **Beware, once an image is erased, it's gone forever!**

## Configuration

- The application saves your preferences (selected folder, hash size, items per page, etc.) between sessions.
- You can toggle the "Keep current preferences" option in the Options menu to control whether preferences are saved on exit.
- You can check "Check subfolders" to check the subfolders inside the folder.
- You can check "Check for image transformations" to check more throughfully all images, rotated/flipped images (slower).
- You can filter a specific image format to check against in "Filter image formats".

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

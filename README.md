# ImageDuplicateChecker

A simple Python application with a GUI (PyQt5) for finding and managing duplicate images in a specified folder (I've adapted it from a script that I use to deduplicate image datasets).

## Features

- Select a folder to scan for duplicate images (Only supports '.png', '.jpg', '.gif', '.bmp', '.tiff' images)
- Adjust hash size for controlling sensitivity of duplicate detection
- Display duplicate images in groups
- Remove selected duplicate images
- Pagination for easier navigation of results
- Progress bar to show scanning progress
- Multithreading + caching of image hashes for improved performance
- Option to check subfolders
- Preferences saving and loading

## Requirements

- Python 3.6+
- PyQt5
- Pillow
- ImageHash

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

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

import sys
import os
import json
import cv2
from PIL import Image, ImageOps
import imagehash
import rawpy
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QSpinBox, QScrollArea, 
                             QCheckBox, QGroupBox, QMessageBox, QComboBox, QInputDialog, QAction, QProgressBar, QDialog, QGridLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize, QThreadPool, QRunnable, pyqtSignal, QObject
import concurrent.futures
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class WorkerSignals(QObject):
    finished = pyqtSignal(tuple)
    progress = pyqtSignal(int, int)  # Emits current progress and total

class DuplicateFinderWorker(QRunnable):
    def __init__(self, folder_path, hash_size, hash_cache, batch_size, check_subfolders, num_threads, image_formats, check_transformations):
        super().__init__()
        self.folder_path = folder_path
        self.hash_size = hash_size
        self.hash_cache = hash_cache
        self.batch_size = batch_size
        self.check_subfolders = check_subfolders
        self.num_threads = num_threads
        self.image_formats = image_formats
        self.check_transformations = check_transformations
        self.signals = WorkerSignals()

    def run(self):
        duplicates, updated_cache = find_similar_images(
            self.folder_path, 
            self.hash_size, 
            self.hash_cache, 
            self.batch_size, 
            self.check_subfolders, 
            self.signals.progress,
            self.num_threads,
            self.image_formats,
            self.check_transformations
        )
        self.signals.finished.emit((duplicates, updated_cache))

class ClickableImageLabel(QLabel):
    def __init__(self, checkbox):
        super().__init__()
        self.checkbox = checkbox

    def mousePressEvent(self, event):
        self.checkbox.setChecked(not self.checkbox.isChecked())

class ImageDuplicateChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.duplicates = []
        self.folder_path = ""
        self.threadpool = QThreadPool()
        self.cache_file = "hash_cache.json"
        self.cache_capacity = 10000 # Nb of elements in the cache
        self.hash_cache = LRUCache(self.cache_capacity)
        self.load_cache()
        self.current_page = 0
        self.items_per_page = 10
        self.batch_size = 100  # Number of images to process in each batch
        self.check_subfolders = False
        self.num_threads = os.cpu_count() or 1  # Default to system CPU count
        self.image_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.ico', '.ppm', '.tga',
                              '.raw', '.arw', '.cr2', '.nef', '.orf', '.rw2', '.dng']
        self.check_transformations = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Duplicate Checker')
        self.setGeometry(100, 100, 800, 600)

        # Create menubar
        menubar = self.menuBar()
        
        # Create Options menu
        optionsMenu = menubar.addMenu('Options')
        
        ## Add actions to the Options menu ##
        self.keep_preferences_action = QAction('Keep current preferences', self, checkable=True)
        self.keep_preferences_action.setChecked(True)
        optionsMenu.addAction(self.keep_preferences_action)

        self.check_subfolders_action = QAction('Check subfolders', self, checkable=True)
        self.check_subfolders_action.setChecked(self.check_subfolders)
        self.check_subfolders_action.triggered.connect(self.toggle_check_subfolders)
        optionsMenu.addAction(self.check_subfolders_action)

        self.check_transformations_action = QAction('Check for image transformations', self, checkable=True)
        self.check_transformations_action.setChecked(self.check_transformations)
        self.check_transformations_action.triggered.connect(self.toggle_check_transformations)
        optionsMenu.addAction(self.check_transformations_action)

        self.set_threads_action = QAction('Set number of threads', self)
        self.set_threads_action.triggered.connect(self.set_num_threads)
        optionsMenu.addAction(self.set_threads_action)

        self.set_batch_size_action = QAction('Set batch size', self)
        self.set_batch_size_action.triggered.connect(self.set_batch_size)
        optionsMenu.addAction(self.set_batch_size_action)

        self.set_cache_size_action = QAction('Set cache size', self)
        self.set_cache_size_action.triggered.connect(self.set_cache_size)
        optionsMenu.addAction(self.set_cache_size_action)

        self.set_image_formats_action = QAction('Filter image formats', self)
        self.set_image_formats_action.triggered.connect(self.show_image_formats_dialog)
        optionsMenu.addAction(self.set_image_formats_action)

        ## Main Widget ##
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls
        controls_layout = QHBoxLayout()
        self.folder_button = QPushButton('Select Folder')
        self.folder_button.clicked.connect(self.select_folder)
        controls_layout.addWidget(self.folder_button)

        # Create a horizontal layout for hash size label and spinbox
        hash_size_layout = QHBoxLayout()
        hash_size_layout.setSpacing(0)  # Remove spacing between label and spinbox
        
        self.hash_size_label = QLabel('Hash Size:')
        hash_size_layout.addWidget(self.hash_size_label)
        self.hash_size_label.setFixedWidth(70)

        self.hash_size_spinbox = QSpinBox()
        self.hash_size_spinbox.setRange(1, 16)
        self.hash_size_spinbox.setValue(6)
        self.hash_size_spinbox.setFixedWidth(60)  # Reduce spinbox width
        hash_size_layout.addWidget(self.hash_size_spinbox)
        
        # Set tooltip for the entire hash_size_layout
        hash_size_widget = QWidget()
        hash_size_widget.setLayout(hash_size_layout)
        hash_size_widget.setToolTip("Lower values increase the sensitivity but may increase the number of false positives.")
        
        # Add the hash size layout to the main controls layout
        controls_layout.addWidget(hash_size_widget)

        self.check_button = QPushButton('Check Duplicates')
        self.check_button.clicked.connect(self.check_duplicates)
        controls_layout.addWidget(self.check_button)

        self.remove_button = QPushButton('Remove Selected')
        self.remove_button.clicked.connect(self.remove_selected)
        controls_layout.addWidget(self.remove_button)

        main_layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Duplicate count labels
        self.duplicate_count_layout = QHBoxLayout()
        self.total_duplicates_label = QLabel()
        self.total_duplicates_label.setVisible(False)  # Hide the label initially
        self.duplicate_count_layout.addWidget(self.total_duplicates_label)
        main_layout.addLayout(self.duplicate_count_layout)

        ## Results area ##
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

        ## Bottom widgets ##
        # Add pagination controls
        pagination_layout = QHBoxLayout()
        self.prev_button = QPushButton('Previous')
        self.prev_button.clicked.connect(self.previous_page)
        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.next_page)
        self.page_label = QLabel('Page 1')
        self.items_per_page_combo = QComboBox()
        self.items_per_page_combo.addItems(['10', '20', '50', '100'])
        self.items_per_page_combo.setCurrentText(str(self.items_per_page))
        self.items_per_page_combo.currentTextChanged.connect(self.change_items_per_page)
        
        pagination_layout.addWidget(self.prev_button)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addWidget(self.next_button)
        pagination_layout.addWidget(QLabel('Unique duplicates per page:'))
        pagination_layout.addWidget(self.items_per_page_combo)
        
        main_layout.addLayout(pagination_layout)

    def toggle_check_subfolders(self):
        self.check_subfolders = self.check_subfolders_action.isChecked()

    def toggle_check_transformations(self):
        self.check_transformations = self.check_transformations_action.isChecked()
        # Empty the LRU cache
        self.hash_cache = LRUCache(self.cache_capacity)

    def closeEvent(self, event):
        if self.keep_preferences_action.isChecked():
            self.save_preferences()
        
        # Remove the cache file if it exists
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                print(f"Cache file '{self.cache_file}' has been removed.")
            except Exception as e:
                print(f"Error removing cache file: {e}")
        
        # Call the parent class closeEvent
        super().closeEvent(event)

    def change_items_per_page(self, value):
        self.items_per_page = int(value)
        self.current_page = 0
        if self.duplicates:  # Only update display if duplicates have been found
            self.display_duplicates()

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_duplicates()

    def next_page(self):
        if (self.current_page + 1) * self.items_per_page < len(self.duplicates):
            self.current_page += 1
            self.display_duplicates()

    def select_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.folder_path:
            self.folder_button.setText(f"Selected: {self.folder_path}")

    def set_num_threads(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Number of Threads")
        dialog.setLabelText("Enter number of threads to use:")
        dialog.setInputMode(QInputDialog.IntInput)
        dialog.setIntRange(1, os.cpu_count() or 1)
        dialog.setIntValue(self.num_threads)
        
        # Remove the '?' button
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        if dialog.exec_() == QInputDialog.Accepted:
            self.num_threads = dialog.intValue()

    def set_batch_size(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Batch Size")
        dialog.setLabelText("Enter batch size for image processing:")
        dialog.setInputMode(QInputDialog.IntInput)
        dialog.setIntRange(1, 1000)
        dialog.setIntValue(self.batch_size)
        
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        if dialog.exec_() == QInputDialog.Accepted:
            self.batch_size = dialog.intValue()

    def set_cache_size(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Set Cache Size")
        dialog.setLabelText("Enter cache size (number of elements):")
        dialog.setInputMode(QInputDialog.IntInput)
        dialog.setIntRange(100, 100000)
        dialog.setIntValue(self.cache_capacity)
        
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        if dialog.exec_() == QInputDialog.Accepted:
            self.cache_capacity = dialog.intValue()
            self.hash_cache = LRUCache(self.cache_capacity)

    def show_image_formats_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Image Formats")
        layout = QGridLayout(dialog)

        all_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.ico', '.ppm', '.tga',
                    '.raw', '.arw', '.cr2', '.nef', '.orf', '.rw2', '.dng']
        checkboxes = {}
        row, col = 0, 0
        for format in all_formats:
            checkbox = QCheckBox(format)
            checkbox.setChecked(format in self.image_formats)
            checkboxes[format] = checkbox
            layout.addWidget(checkbox, row, col)
            col += 1
            if col == 3:  # 3 columns
                col = 0
                row += 1

        def on_accept():
            self.image_formats = [format for format, checkbox in checkboxes.items() if checkbox.isChecked()]
            dialog.accept()

        accept_button = QPushButton("Apply")
        accept_button.clicked.connect(on_accept)
        layout.addWidget(accept_button, row + 1, 0, 1, 3)

        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.exec_()

    def save_preferences(self):
        preferences = {
            'folder_path': self.folder_path,
            'hash_size': self.hash_size_spinbox.value(),
            'items_per_page': self.items_per_page,
            'check_subfolders': self.check_subfolders,
            'num_threads': self.num_threads,
            'batch_size': self.batch_size,
            'cache_capacity': self.cache_capacity,
            'image_formats': self.image_formats,
            'check_transformations': self.check_transformations
        }
        with open('preferences.json', 'w') as f:
            json.dump(preferences, f)

    def load_preferences(self):
        try:
            with open('preferences.json', 'r') as f:
                preferences = json.load(f)
                self.folder_path = preferences.get('folder_path', '')
                self.folder_button.setText(f"Selected: {self.folder_path}" if self.folder_path else "Select Folder")
                self.hash_size_spinbox.setValue(preferences.get('hash_size', 6))
                self.items_per_page = preferences.get('items_per_page', 10)
                self.items_per_page_combo.setCurrentText(str(self.items_per_page))
                self.check_subfolders = preferences.get('check_subfolders', False)
                self.check_subfolders_action.setChecked(self.check_subfolders)
                self.num_threads = preferences.get('num_threads', os.cpu_count() or 1)
                self.batch_size = preferences.get('batch_size', 100)
                self.cache_capacity = preferences.get('cache_capacity', 10000)
                self.hash_cache = LRUCache(self.cache_capacity)
                loaded_formats = preferences.get('image_formats', self.image_formats)
                self.image_formats = loaded_formats
                self.check_transformations = preferences.get('check_transformations', False)
                self.check_transformations_action.setChecked(self.check_transformations)
        except FileNotFoundError:
            pass
    
    def check_duplicates(self):
        if not self.folder_path:
            return

        hash_size = self.hash_size_spinbox.value()
        worker = DuplicateFinderWorker(self.folder_path, hash_size, self.hash_cache, self.batch_size, self.check_subfolders, self.num_threads, self.image_formats, self.check_transformations)
        worker.signals.finished.connect(self.on_duplicates_found)
        worker.signals.progress.connect(self.update_progress)
        self.threadpool.start(worker)

        # Show and reset the progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_duplicates_found(self, result):
        self.duplicates, self.hash_cache = result
        self.save_cache()
        
        # Hide the progress bar
        self.progress_bar.setVisible(False)

        # Calculate and display duplicate counts
        unique_duplicates = len(self.duplicates)
        total_duplicates = sum(len(group) for group in self.duplicates)
        self.total_duplicates_label.setText(f"Total duplicates: {total_duplicates} / Unique duplicates: {unique_duplicates}")
        self.total_duplicates_label.setVisible(True)  # Show the label
        
        # Print to terminal
        print(f"Done. Found {total_duplicates} total duplicates in {unique_duplicates} unique groups.")

        self.current_page = 0
        self.display_duplicates()

    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                for key, value in cache_data.items():
                    self.hash_cache.put(key, value)  # Store the hash as a string
        except FileNotFoundError:
            pass

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(dict(self.hash_cache.cache), f)

    def display_duplicates(self):
        # Clear previous results
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.duplicates:
            self.scroll_layout.addWidget(QLabel("No duplicates found."))
            self.update_pagination_controls()
            return

        start_index = self.current_page * self.items_per_page
        end_index = min(start_index + self.items_per_page, len(self.duplicates))


        for dup_group in self.duplicates[start_index:end_index]:
            group_box = QGroupBox()
            group_layout = QHBoxLayout(group_box)
            group_layout.setSpacing(10)
            group_layout.setContentsMargins(10, 10, 10, 10)

            for img_path in dup_group:
                img_widget = QWidget()
                img_widget.setFixedSize(220, 280)
                img_layout = QVBoxLayout(img_widget)
                img_layout.setContentsMargins(0, 0, 0, 0)

                img_info = self.get_image_info(img_path)
                
                checkbox = QCheckBox(f"{os.path.basename(img_path)}\n{img_info}")
                checkbox.setStyleSheet("QCheckBox { padding: 5px; }")
                checkbox.setProperty("full_path", img_path)  # Store the full path as a property
                
                pixmap = QPixmap(img_path)
                pixmap = pixmap.scaled(QSize(200, 200), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label = ClickableImageLabel(checkbox)
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                img_layout.addWidget(img_label)

                img_layout.addWidget(checkbox)

                group_layout.addWidget(img_widget)

            group_layout.addStretch(1)
            self.scroll_layout.addWidget(group_box)

        self.scroll_layout.addStretch(1)
        self.update_pagination_controls()

    def update_pagination_controls(self):
        total_pages = (len(self.duplicates) + self.items_per_page - 1) // self.items_per_page
        self.page_label.setText(f'Page {self.current_page + 1} of {total_pages}')
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled((self.current_page + 1) * self.items_per_page < len(self.duplicates))

    def get_image_info(self, img_path):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
            file_size = os.path.getsize(img_path)
            size_kb = file_size / 1024
            return f"{width}x{height}, {size_kb:.1f} KB"
        except Exception as e:
            print(f"Error getting image info for {img_path}: {e}")
            return "Unknown"

    def remove_selected(self):
        selected_count = 0
        selected_files = []

        for i in range(self.scroll_layout.count()):
            group_box = self.scroll_layout.itemAt(i).widget()
            if isinstance(group_box, QGroupBox):
                for j in range(group_box.layout().count()):
                    img_widget = group_box.layout().itemAt(j).widget()
                    if isinstance(img_widget, QWidget):
                        checkbox = img_widget.layout().itemAt(1).widget()
                        if checkbox.isChecked():
                            selected_count += 1
                            img_path = checkbox.property("full_path")  # Get the full path from the property
                            selected_files.append(img_path)

        if selected_count == 0:
            QMessageBox.information(self, "No Selection", "No images selected for removal.")
            return

        if selected_count > 1:
            confirm = QMessageBox.question(self, "Confirm Removal", 
                                        f"Are you sure you want to remove {selected_count} selected images?",
                                        QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.No:
                return

        # Show and reset the progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(selected_files))
        self.progress_bar.setValue(0)

        for i, img_path in enumerate(selected_files):
            try:
                os.remove(img_path)
                print(f"Removed: {img_path}")
            except Exception as e:
                print(f"Error removing {img_path}: {e}")
            self.progress_bar.setValue(i + 1)
        
        # Hide the progress bar
        self.progress_bar.setVisible(False)

        # Refresh the display
        self.check_duplicates()

def is_valid_image(file_path, image_formats):
    return file_path.lower().endswith(tuple(image_formats))

def open_image(image_path):
    raw_extensions = ('.raw', '.arw', '.cr2', '.nef', '.orf', '.rw2', '.dng')
    if image_path.lower().endswith(raw_extensions):
        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess()
        return Image.fromarray(rgb)
    else:
        return Image.open(image_path)

def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img.resize((256, 256), Image.Resampling.LANCZOS)

def compute_color_moment_hash(img, hash_size=8):
    img_array = np.array(img.resize((hash_size, hash_size)))
    means = np.mean(img_array, axis=(0, 1))
    variances = np.var(img_array, axis=(0, 1))
    skewness = np.mean(((img_array - means) / variances) ** 3, axis=(0, 1))
    
    moments = np.concatenate([means, variances, skewness])
    hash_bits = (moments > np.median(moments)).astype(int)
    
    # Ensure the hash is the correct size (hash_size * hash_size)
    hash_bits = np.resize(hash_bits, (hash_size, hash_size))
    return imagehash.ImageHash(hash_bits)

def compute_edge_hash(img, hash_size=8):
    img_array = np.array(img.convert('L').resize((hash_size, hash_size)))
    edges = cv2.Canny(img_array, 100, 200)
    
    # Ensure the hash is the correct size (hash_size * hash_size)
    edges = np.resize(edges, (hash_size, hash_size))
    return imagehash.ImageHash(edges > 0)

def compute_hashes(img, hash_size=8):
    p_hash = imagehash.phash(img, hash_size=hash_size)
    d_hash = imagehash.dhash(img, hash_size=hash_size)
    color_hash = compute_color_moment_hash(img, hash_size=hash_size)
    edge_hash = compute_edge_hash(img, hash_size=hash_size)
    return p_hash, d_hash, color_hash, edge_hash

def combine_hashes(phash, dhash, color_hash, edge_hash, weights=(0.5, 0.3, 0.1, 0.1)):
    p_array = np.array(phash.hash.flatten().astype(int))
    d_array = np.array(dhash.hash.flatten().astype(int))
    c_array = np.array(color_hash.hash.flatten().astype(int))
    e_array = np.array(edge_hash.hash.flatten().astype(int))
    
    combined = (weights[0] * p_array + weights[1] * d_array + 
                weights[2] * c_array + weights[3] * e_array)
    combined = (combined > 0.5).astype(int)
    
    return imagehash.ImageHash(combined.reshape((int(np.sqrt(len(combined))), -1)))

def process_image(file_path, hash_size, hash_cache):
    try:
        cache_key = f"{file_path}_{hash_size}"
        cached_hash = hash_cache.get(cache_key)
        if cached_hash:
            return imagehash.hex_to_hash(cached_hash), file_path

        with open_image(file_path) as img:
            img = preprocess_image(img)
            p_hash, d_hash, color_hash, edge_hash = compute_hashes(img, hash_size)
            combined_hash = combine_hashes(p_hash, d_hash, color_hash, edge_hash)
        
        hash_cache.put(cache_key, str(combined_hash))
        return combined_hash, file_path
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_image_transformations(image_path, hash_size, hash_cache):
    try:
        cache_key = f"{image_path}_{hash_size}"
        cached_hash = hash_cache.get(cache_key)
        if cached_hash:
            return imagehash.hex_to_hash(cached_hash), image_path

        with open_image(image_path) as img:
            img = preprocess_image(img)

            transformations = [
                lambda x: x,  # Original image
                ImageOps.mirror,
                ImageOps.flip,
                ImageOps.exif_transpose,
                lambda x: x.rotate(90),   # 90 degrees rotation
                lambda x: x.rotate(180),  # 180 degrees rotation
                lambda x: x.rotate(270)   # 270 degrees rotation
            ]

            hashes = [compute_hashes(transform(img), hash_size) for transform in transformations]
            valid_hashes = [h for h in hashes if all(x is not None for x in h)]
            
            if not valid_hashes:
                raise ValueError("No valid hashes computed")
            
            combined_hashes = [combine_hashes(*h) for h in valid_hashes]
            
            # Use a strict consensus hash as the final hash
            consensus_hash = get_consensus_hash(combined_hashes, threshold=0.8)
            
            hash_cache.put(cache_key, str(consensus_hash))

        return consensus_hash, image_path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_consensus_hash(hashes, threshold=0.8):
    # Convert hashes to binary arrays
    binary_hashes = [np.array(h.hash).flatten() for h in hashes]
    
    # Calculate the mean for each bit position
    mean_bits = np.mean(binary_hashes, axis=0)
    
    # Apply strict consensus: 1 if >= threshold, 0 if <= (1-threshold), -1 otherwise
    consensus_bits = np.where(mean_bits >= threshold, 1, 
                              np.where(mean_bits <= (1-threshold), 0, -1))
    
    # Replace any uncertain bits (-1) with the original image's bits
    original_bits = binary_hashes[0]
    consensus_bits = np.where(consensus_bits == -1, original_bits, consensus_bits)
    
    # Reshape back to square and create ImageHash object
    hash_size = int(np.sqrt(len(consensus_bits)))
    return imagehash.ImageHash(consensus_bits.reshape(hash_size, hash_size))

def find_similar_images(folder_path, hash_size=8, hash_cache=None, batch_size=100, check_subfolders=False, progress_callback=None, num_threads=None, image_formats=None, check_transformations=False):
    if hash_cache is None:
        hash_cache = LRUCache(10000)
    
    hashes = {}
    image_files = []
    
    def scan_directory(dir_path):
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file():
                    if is_valid_image(entry.path, image_formats):
                        image_files.append(entry.path)
                    else:
                        print(f"Skipping non-image file: {entry.path}")
                elif check_subfolders and entry.is_dir():
                    scan_directory(entry.path)
    
    scan_directory(folder_path)
    
    total_images = len(image_files)
    processed_images = 0
    
    process_func = process_image_transformations if check_transformations else process_image
    
    while processed_images < total_images:
        batch = image_files[processed_images:processed_images + batch_size]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_func, file_path, hash_size, hash_cache) for file_path in batch]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    file_hash, file_path = result
                    if file_hash in hashes:
                        hashes[file_hash].append(file_path)
                    else:
                        hashes[file_hash] = [file_path]
        
        processed_images += len(batch)
        print(f"Processed {processed_images}/{total_images} images")
        if progress_callback:
            progress_callback.emit(processed_images, total_images)
    
    duplicates = [tuple(file_paths) for file_paths in hashes.values() if len(file_paths) > 1]
    return duplicates, hash_cache

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageDuplicateChecker()
    ex.load_preferences()  # Load preferences when starting the application
    ex.show()
    sys.exit(app.exec_())

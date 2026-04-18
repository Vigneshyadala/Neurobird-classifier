"""
Downloads and organizes the CUB-200-2011 Bird Dataset
This is a well-known dataset with 200 bird species
"""

import urllib.request
import tarfile
import os
import shutil
from pathlib import Path

# Dataset URL
DATASET_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
DOWNLOAD_FILE = "CUB_200_2011.tgz"
EXTRACT_DIR = "CUB_200_2011"
OUTPUT_DIR = "test-data"

def download_dataset():
    """Download the CUB-200 dataset"""
    
    print("=" * 60)
    print("CUB-200-2011 Bird Dataset Downloader")
    print("=" * 60)
    print(f"\nDataset: 200 bird species, ~12,000 images")
    print(f"Size: ~1.1 GB")
    print(f"Downloading from: {DATASET_URL}\n")
    
    if os.path.exists(DOWNLOAD_FILE):
        print(f"✓ Dataset file already exists: {DOWNLOAD_FILE}")
        return True
    
    try:
        print("Downloading... (this may take several minutes)")
        print("Progress: ", end="", flush=True)
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if count % 50 == 0:  # Update every 50 blocks
                print(f"{percent}%...", end="", flush=True)
        
        urllib.request.urlretrieve(DATASET_URL, DOWNLOAD_FILE, progress_hook)
        print("\n✓ Download complete!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def extract_dataset():
    """Extract the downloaded dataset"""
    
    print("Extracting dataset...")
    
    if not os.path.exists(DOWNLOAD_FILE):
        print(f"✗ Dataset file not found: {DOWNLOAD_FILE}")
        return False
    
    try:
        with tarfile.open(DOWNLOAD_FILE, 'r:gz') as tar:
            tar.extractall('.')
        print("✓ Extraction complete!\n")
        return True
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def organize_images():
    """Organize images into species folders"""
    
    print("Organizing images by species...")
    
    images_dir = Path(EXTRACT_DIR) / "images"
    
    if not images_dir.exists():
        print(f"✗ Images directory not found: {images_dir}")
        return False
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    species_count = 0
    total_images = 0
    
    # Each folder in images/ is a species
    for species_folder in sorted(images_dir.iterdir()):
        if not species_folder.is_dir():
            continue
        
        # Get species name (remove number prefix)
        species_name = species_folder.name.split('.', 1)[1] if '.' in species_folder.name else species_folder.name
        species_name = species_name.replace('_', ' ')
        
        # Create species folder in output
        output_species_dir = output_path / species_name
        output_species_dir.mkdir(exist_ok=True)
        
        # Copy images
        image_count = 0
        for img_file in species_folder.glob('*.jpg'):
            shutil.copy2(img_file, output_species_dir / img_file.name)
            image_count += 1
        
        species_count += 1
        total_images += image_count
        
        if species_count % 20 == 0:
            print(f"  Processed {species_count} species...")
    
    print(f"✓ Organization complete!")
    print(f"  - Species: {species_count}")
    print(f"  - Total images: {total_images}")
    print(f"  - Location: {OUTPUT_DIR}/\n")
    
    return True

def cleanup():
    """Optional: Clean up extracted files"""
    
    print("\nCleanup options:")
    print("1. Keep all files (recommended)")
    print("2. Delete compressed file only (.tgz)")
    print("3. Delete extracted folder only")
    print("4. Delete both")
    
    choice = input("\nEnter choice (1-4) or press Enter to skip: ").strip()
    
    if choice == "2":
        if os.path.exists(DOWNLOAD_FILE):
            os.remove(DOWNLOAD_FILE)
            print(f"✓ Deleted {DOWNLOAD_FILE}")
    elif choice == "3":
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)
            print(f"✓ Deleted {EXTRACT_DIR}/")
    elif choice == "4":
        if os.path.exists(DOWNLOAD_FILE):
            os.remove(DOWNLOAD_FILE)
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)
        print(f"✓ Cleaned up all temporary files")
    else:
        print("Keeping all files")

def main():
    """Main execution"""
    
    # Step 1: Download
    if not download_dataset():
        print("\nFailed to download dataset. Please check your internet connection.")
        return
    
    # Step 2: Extract
    if not extract_dataset():
        print("\nFailed to extract dataset.")
        return
    
    # Step 3: Organize
    if not organize_images():
        print("\nFailed to organize images.")
        return
    
    # Step 4: Cleanup (optional)
    cleanup()
    
    # Final instructions
    print("\n" + "=" * 60)
    print("SUCCESS! Your dataset is ready!")
    print("=" * 60)
    print(f"\nYour bird images are organized in: {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("1. Run: python organize_images.py")
    print("   (This will split into train/validation folders)")
    print("2. Then run: python train.py")
    print("   (This will train your bird classifier)")
    print("\nHappy training! 🐦")

if __name__ == "__main__":
    main()
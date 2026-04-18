import os
import shutil
from pathlib import Path
import random

def organize_images():
    """Organize CUB-200 images into train/validation split"""
    
    # Paths
    source_dir = Path('CUB_200_2011/images')
    train_dir = Path('bird-dataset/train')
    val_dir = Path('bird-dataset/validation')
    
    # Create directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if source exists
    if not source_dir.exists():
        print(f"Error: {source_dir} not found!")
        return
    
    # Get all species folders
    species_folders = [f for f in source_dir.iterdir() if f.is_dir()]
    
    if len(species_folders) == 0:
        print("No species folders found!")
        return
    
    print(f"Found {len(species_folders)} species folders")
    print("Organizing images into train/validation split (75%/25%)...")
    
    train_count = 0
    val_count = 0
    
    for idx, species_folder in enumerate(species_folders, 1):
        species_name = species_folder.name
        
        # Get all images in this species folder
        images = list(species_folder.glob('*.jpg'))
        
        if len(images) == 0:
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Split 75% train, 25% validation
        split_idx = int(len(images) * 0.75)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create species folders in train and validation
        train_species_dir = train_dir / species_name
        val_species_dir = val_dir / species_name
        
        train_species_dir.mkdir(exist_ok=True)
        val_species_dir.mkdir(exist_ok=True)
        
        # Copy training images
        for img in train_images:
            shutil.copy2(img, train_species_dir / img.name)
            train_count += 1
        
        # Copy validation images
        for img in val_images:
            shutil.copy2(img, val_species_dir / img.name)
            val_count += 1
        
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(species_folders)} species...")
    
    print("\n" + "="*60)
    print("SUCCESS! Dataset organization complete!")
    print("="*60)
    print(f"\nTrain images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Total species: {len(species_folders)}")
    print(f"\nDataset location: bird-dataset/")
    print("  - bird-dataset/train/")
    print("  - bird-dataset/validation/")
    print("\nNext step: Run 'python train.py' to train your model!")

if __name__ == "__main__":
    organize_images()
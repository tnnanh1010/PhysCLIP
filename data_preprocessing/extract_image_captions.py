import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any


def extract_images_from_filtered_file(filtered_file_path: str) -> List[Dict[str, str]]:
    """
    Extract image paths and descriptions from a single filtered JSON file.
    
    Args:
        filtered_file_path: Path to the filtered JSON file
    
    Returns:
        list: List of dictionaries with image_path and description
    """
    try:
        with open(filtered_file_path, 'r') as f:
            data = json.load(f)
        
        extracted_images = []
        for image in data.get("images", []):
            extracted_images.append({
                "image_path": image.get("full_image_path", ""),
                "description": image.get("description", "")
            })
        
        return extracted_images
    
    except Exception as e:
        print(f"Error processing {filtered_file_path}: {str(e)}")
        return []


def extract_all_images_to_jsonl(
    base_path: str = "/bigdisk/minhpvt/quick-test/na/ML/data/batches",
    batch_start: int = 1,
    batch_end: int = 200,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Extract all image paths and descriptions from filtered files across batches
    and save to a JSONL file.
    
    Args:
        base_path: Base path to the batches folder
        batch_start: Starting batch number (inclusive)
        batch_end: Ending batch number (inclusive)
        output_file: Path to output JSONL file (optional)
    
    Returns:
        dict: Extraction results and metadata
    """
    all_images = []
    total_files_processed = 0
    total_files_found = 0
    batch_stats = {}
    
    print(f"Extracting images from batches {batch_start} to {batch_end}")
    
    for batch_num in range(batch_start, batch_end + 1):
        batch_folder = f"batch_{batch_num:04d}"
        batch_path = os.path.join(base_path, batch_folder)
        papers_path = os.path.join(batch_path, "papers")
        
        if not os.path.exists(papers_path):
            continue
        
        # Find all filtered JSON files in this batch
        filtered_files = glob.glob(os.path.join(papers_path, "**/*_filtered.json"), recursive=True)
        
        if not filtered_files:
            continue
        
        batch_images = []
        for filtered_file in filtered_files:
            images = extract_images_from_filtered_file(filtered_file)
            batch_images.extend(images)
            total_files_processed += 1
        
        total_files_found += len(filtered_files)
        all_images.extend(batch_images)
        
        batch_stats[batch_folder] = {
            "filtered_files": len(filtered_files),
            "images_extracted": len(batch_images)
        }
        
        print(f"  {batch_folder}: {len(filtered_files)} files, {len(batch_images)} images")
    
    # Generate output file path if not provided
    if output_file is None:
        output_file = os.path.join(base_path, f"image_captions_{batch_start}_{batch_end}.jsonl")
    
    # Write to JSONL file
    with open(output_file, 'w') as f:
        for image_data in all_images:
            f.write(json.dumps(image_data) + '\n')
    
    results = {
        "total_batches_scanned": batch_end - batch_start + 1,
        "total_filtered_files_found": total_files_found,
        "total_filtered_files_processed": total_files_processed,
        "total_images_extracted": len(all_images),
        "output_file": output_file,
        "batch_stats": batch_stats
    }
    
    # Also save metadata as JSON
    metadata_file = output_file.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Batches scanned: {results['total_batches_scanned']}")
    print(f"Filtered files found: {results['total_filtered_files_found']}")
    print(f"Filtered files processed: {results['total_filtered_files_processed']}")
    print(f"Total images extracted: {results['total_images_extracted']}")
    print(f"\nOutput saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")
    
    return results


if __name__ == "__main__":
    # Extract from all batches
    results = extract_all_images_to_jsonl(
        batch_start=1,
        batch_end=34
    )

import asyncio
import json
import os
import glob
from pathlib import Path
import datetime
from typing import Dict, List, Any

async def filter_single_paper(
    input_file_path: str,
    content_threshold: int,
    output_file_path: str = None
) -> Dict[str, Any]:
    """
    Filter a single paper based on image content threshold.
    
    Args:
        input_file_path: Path to the scored JSON file
        content_threshold: Minimum score for individual images
        output_file_path: Path to save filtered results (optional)
    
    Returns:
        dict: Filtering results and metadata
    """
    try:
        # Load the scored paper data
        with open(input_file_path, 'r') as f:
            paper_data = json.load(f)
        
        # Get original counts (images only)
        original_counts = {
            "images": len(paper_data.get("images", []))
        }
        
        # Filter images based on content threshold AND description indicating a figure
        filtered_images = []
        for image in paper_data.get("images", []):
            score = image.get("astro_relevance_score", 0)
            description = image.get("description", "").strip()
            # Check if description starts with "Figure" or "Fig." (case-insensitive)
            is_figure = description.lower().startswith(("figure", "fig."))
            # Check if description contains "same as" (case-insensitive)
            has_same_as = "same as" in description.lower()
            
            if score >= content_threshold and is_figure and not has_same_as:
                # Extract only the caption text after "Figure X:" or "Fig. X." or "Figure X."
                caption = description
                
                if description.lower().startswith("figure"):
                    # Handle "Figure X:" or "Figure X." format
                    # Search for first colon or period after "Figure "
                    start_idx = 6  # After "Figure"
                    colon_idx = description.find(":", start_idx)
                    period_idx = description.find(".", start_idx)
                    
                    # Use whichever comes first (colon or period)
                    if colon_idx != -1 and (period_idx == -1 or colon_idx < period_idx):
                        caption = description[colon_idx + 1:].strip()
                    elif period_idx != -1:
                        caption = description[period_idx + 1:].strip()
                        
                elif description.lower().startswith("fig."):
                    # Handle "Fig. X." or "Fig. X:" format
                    first_period = description.find(".", 3)  # First period after "Fig"
                    if first_period != -1:
                        # Look for colon or second period after figure number
                        rest = description[first_period + 1:].strip()
                        colon_idx = rest.find(":")
                        period_idx = rest.find(".")
                        
                        if colon_idx != -1 and (period_idx == -1 or colon_idx < period_idx):
                            caption = rest[colon_idx + 1:].strip()
                        elif period_idx != -1:
                            caption = rest[period_idx + 1:].strip()
                        else:
                            caption = rest
                
                # Check if caption length is greater than 20 characters
                if len(caption) > 20:
                    # Create a copy of the image with updated description
                    filtered_image = image.copy()
                    filtered_image["description"] = caption
                    filtered_image["original_description"] = description  # Keep original for reference
                    filtered_images.append(filtered_image)
        
        # Create filtered paper data
        filtered_paper_data = paper_data.copy()
        filtered_paper_data["images"] = filtered_images
        
        # Update component counts
        filtered_counts = {
            "images": len(filtered_images)
        }
        
        filtered_paper_data["component_count"] = {
            **filtered_counts,
            "total_components": filtered_counts["images"]
        }
        
        # Add filtering metadata
        filtered_paper_data["filtering_metadata"] = {
            "content_threshold": content_threshold,
            "figure_description_only": True,  # Only keeps images with Figure/Fig. descriptions
            "original_counts": original_counts,
            "filtered_counts": filtered_counts,
            "images_removed": original_counts["images"] - filtered_counts["images"],
            "filtering_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Generate output file path if not provided
        if output_file_path is None:
            base_path = input_file_path.rsplit('_scored.json', 1)[0]
            output_file_path = f"{base_path}_filtered.json"
        
        # Save filtered data
        with open(output_file_path, 'w') as f:
            json.dump(filtered_paper_data, f, indent=2)
        
        return {
            "status": "success",
            "original_file": input_file_path,
            "filtered_file": output_file_path,
            "original_counts": original_counts,
            "filtered_counts": filtered_counts,
            "images_removed": original_counts["images"] - filtered_counts["images"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "original_file": input_file_path,
            "filtered_file": None,
            "original_counts": {"images": 0},
            "filtered_counts": {"images": 0},
            "images_removed": 0
        }

async def filter_multiple_papers_parallel(
    paper_folder_path: str,
    content_threshold: int,
    max_concurrent_papers: int = 10
) -> List[Dict[str, Any]]:
    """
    Filter multiple papers in parallel.
    
    Args:
        paper_folder_path: Path to folder containing scored JSON files
        content_threshold: Minimum score for individual images
        max_concurrent_papers: Maximum number of papers to process concurrently
    
    Returns:
        list: List of filtering results
    """
    # Find all scored JSON files
    scored_files = glob.glob(os.path.join(paper_folder_path, "**/*_scored.json"), recursive=True)
    
    if not scored_files:
        print(f"No scored JSON files found in {paper_folder_path}")
        return []
    
    print(f"Found {len(scored_files)} scored files to filter")
    
    # Process papers in parallel with semaphore control
    semaphore = asyncio.Semaphore(max_concurrent_papers)
    
    async def filter_single_paper_with_logging(scored_file):
        async with semaphore:
            paper_name = os.path.basename(scored_file)
            print(f"  Filtering: {paper_name}")
            try:
                result = await filter_single_paper(
                    scored_file,
                    content_threshold
                )
                
                if result["status"] == "success":
                    print(f"  ✓ Filtered: {paper_name} - Kept {result['filtered_counts']['images']} images (removed {result['images_removed']})")
                else:
                    print(f"  ✗ Error: {paper_name}")
                
                return result
            except Exception as e:
                print(f"  ✗ Failed: {paper_name} - {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "original_file": scored_file,
                    "filtered_file": None
                }
    
    # Create tasks for all papers
    tasks = [filter_single_paper_with_logging(scored_file) for scored_file in scored_files]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    processed_results = []
    for result in results:
        if isinstance(result, dict):
            processed_results.append(result)
        else:
            processed_results.append({
                "status": "error",
                "error": str(result),
                "original_file": "unknown",
                "filtered_file": None
            })
    
    return processed_results

async def filtering_multiple_batches_parallel(
    base_path: str = "/bigdisk/minhpvt/quick-test/na/ML/data/batches",
    batch_start: int = 1,
    batch_end: int = 200,
    content_threshold: int = 3,
    max_concurrent_papers: int = 10
) -> Dict[str, Any]:
    """
    Filter multiple batches of papers in parallel (images only).
    
    Args:
        base_path: Base path to the batches folder
        batch_start: Starting batch number (inclusive)
        batch_end: Ending batch number (inclusive)
        content_threshold: Minimum score for individual images (0-5)
        max_concurrent_papers: Maximum number of papers to process concurrently per batch
    
    Returns:
        dict: Complete filtering results and metadata
    """
    total_papers_processed = 0
    total_papers_success = 0
    total_papers_failed = 0
    total_images_original = 0
    total_images_kept = 0
    batch_results = {}
    all_filtered_files = []
    
    print(f"Starting batch filtering from batch {batch_start} to {batch_end}")
    print(f"Content threshold: {content_threshold}")
    print(f"Max concurrent papers per batch: {max_concurrent_papers}")
    
    for batch_num in range(batch_start, batch_end + 1):
        batch_folder = f"batch_{batch_num:04d}"
        batch_path = os.path.join(base_path, batch_folder)
        papers_path = os.path.join(batch_path, "papers")
        
        print(f"\n{'='*50}")
        print(f"Filtering {batch_folder}")
        print(f"{'='*50}")
        
        # Check if batch folder exists
        if not os.path.exists(papers_path):
            print(f"Papers folder not found: {papers_path}")
            batch_results[batch_folder] = {
                "status": "folder_not_found",
                "papers_processed": 0,
                "papers_kept": 0,
                "papers_filtered_out": 0,
                "papers_failed": 0
            }
            continue
        
        try:
            batch_start_time = asyncio.get_event_loop().time()
            
            # Filter papers in this batch
            results = await filter_multiple_papers_parallel(
                papers_path,
                content_threshold,
                max_concurrent_papers
            )
            
            # Process batch results
            batch_processed = len(results)
            batch_success = sum(1 for r in results if r["status"] == "success")
            batch_failed = sum(1 for r in results if r["status"] == "error")
            batch_images_original = sum(r.get("original_counts", {}).get("images", 0) for r in results)
            batch_images_kept = sum(r.get("filtered_counts", {}).get("images", 0) for r in results)
            
            # Collect filtered file paths
            batch_filtered_files = [r["filtered_file"] for r in results if r["filtered_file"]]
            all_filtered_files.extend(batch_filtered_files)
            
            batch_end_time = asyncio.get_event_loop().time()
            batch_duration = batch_end_time - batch_start_time
            
            total_papers_processed += batch_processed
            total_papers_success += batch_success
            total_papers_failed += batch_failed
            total_images_original += batch_images_original
            total_images_kept += batch_images_kept
            
            batch_results[batch_folder] = {
                "status": "completed",
                "papers_processed": batch_processed,
                "papers_success": batch_success,
                "papers_failed": batch_failed,
                "images_original": batch_images_original,
                "images_kept": batch_images_kept,
                "images_removed": batch_images_original - batch_images_kept,
                "duration_seconds": batch_duration,
                "filtered_files": batch_filtered_files
            }
            
            print(f"\n{batch_folder} Summary:")
            print(f"  Processed: {batch_processed}")
            print(f"  ✓ Success: {batch_success}")
            print(f"  ⚠ Failed: {batch_failed}")
            print(f"  Images: {batch_images_kept}/{batch_images_original} kept")
            print(f"  Duration: {batch_duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error filtering batch {batch_folder}: {str(e)}")
            batch_results[batch_folder] = {
                "status": "batch_failed",
                "papers_processed": 0,
                "papers_success": 0,
                "papers_failed": 0,
                "images_original": 0,
                "images_kept": 0,
                "error": str(e)
            }
    
    # Create comprehensive results
    filtering_results = {
        "filtering_metadata": {
            "batch_start": batch_start,
            "batch_end": batch_end,
            "content_threshold": content_threshold,
            "figure_description_only": True,  # Only keeps images with Figure/Fig. descriptions
            "max_concurrent_papers": max_concurrent_papers,
            "filtering_timestamp": datetime.datetime.now().isoformat()
        },
        "summary": {
            "total_papers_processed": total_papers_processed,
            "total_papers_success": total_papers_success,
            "total_papers_failed": total_papers_failed,
            "total_images_original": total_images_original,
            "total_images_kept": total_images_kept,
            "total_images_removed": total_images_original - total_images_kept,
            "image_keep_rate": total_images_kept / total_images_original * 100 if total_images_original > 0 else 0
        },
        "batch_details": batch_results,
    }
    
    # Save filtering results
    results_file = os.path.join(
        base_path,
        f"filtering_results_{batch_start}_{batch_end}_ct{content_threshold}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(results_file, 'w') as f:
        json.dump(filtering_results, f, indent=2)
    
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Batches processed: {batch_end - batch_start + 1}")
    print(f"Total papers processed: {total_papers_processed}")
    print(f"Papers success: {total_papers_success}")
    print(f"Papers failed: {total_papers_failed}")
    print(f"Total images: {total_images_original}")
    print(f"Images kept: {total_images_kept} ({total_images_kept / total_images_original * 100:.1f}%)" if total_images_original > 0 else "Images kept: 0")
    print(f"Images removed: {total_images_original - total_images_kept}")
    print(f"Filtered files created: {len(all_filtered_files)}")
    print(f"\nResults saved to: {results_file}")
    
    return filtering_results

if __name__ == "__main__":
    # Example usage
    loop = asyncio.get_event_loop()

    # Filter one paper
    # results = loop.run_until_complete(
    #     filter_single_paper(
    #         input_file_path="/path/to/paper_scored.json",
    #         content_threshold=3,
    #         output_file_path="/path/to/paper_filtered.json"
    #     )
    # )
    # print("Single paper filtering result:", results)
    
    # Filter batches with specific thresholds
    results = loop.run_until_complete(
        filtering_multiple_batches_parallel(
            batch_start=1,
            batch_end=34,
            content_threshold=3,  # Images must score 3+ to be kept
            max_concurrent_papers=400
        )
    )
    
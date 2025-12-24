import asyncio
import json
import os
import glob
import datetime
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Import the VLLM functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from vllm_server.vllm_chat_completion import VLLMModelRunner

# CLIP Caption Generation Prompt
CLIP_CAPTION_PROMPT = """You are an expert at creating concise, descriptive captions for scientific images. Your task is to generate a high-quality caption suitable for training a CLIP (Contrastive Language-Image Pre-training) model.

Given the image and its context, create a caption that:
1. Is concise but descriptive in one sentence
2. Describes the visual content accurately and specifically
3. Includes relevant scientific terminology when appropriate
4. Focuses on what is visually present in the image
5. Avoids vague phrases like "this image shows" or "a picture of"
6. Captures the key scientific concepts or data depicted

Original description and context:
{content}

Based on your analysis of the image and context, generate a CLIP-optimized caption.
Respond with ONLY the caption text, nothing else.
"""

async def generate_clip_caption(vlm_runner: VLLMModelRunner, description: str, context_paragraphs: List[str], image_path: str) -> Dict[str, Any]:
    """Generate CLIP-optimized caption for an image using VLM."""
    try:
        # Combine description and context
        text_content = description + "\n" + "\n".join(context_paragraphs)
        
        # Check if image exists
        if os.path.exists(image_path):
            # Convert to file:// URL for local images
            image_url = f"file://{os.path.abspath(image_path)}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CLIP_CAPTION_PROMPT.format(content=text_content)},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
            
            response = await vlm_runner.run_model_completion(messages, model_type="vlm")
            clip_caption = vlm_runner.extract_message_content(response)
            
            # Clean up the caption (remove any extra formatting)
            clip_caption = clip_caption.strip()
            
            return {
                "status": "success",
                "clip_caption": clip_caption,
                "original_description": description,
                "original_context": text_content,
                "image_path": image_path,
                "used_image": True
            }
        else:
            # Image not found
            return {
                "status": "error",
                "error": f"Image not found: {image_path}",
                "clip_caption": "",
                "original_description": description,
                "original_context": text_content,
                "image_path": image_path,
                "used_image": False
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "clip_caption": "",
            "original_description": description,
            "original_context": text_content if 'text_content' in locals() else description,
            "image_path": image_path,
            "used_image": False
        }

async def generate_clip_captions_for_paper(
    input_file_path: str,
    output_file_path: str = None
) -> Dict[str, Any]:
    """
    Generate CLIP captions for all images in a single paper.
    """
    try:
        # Initialize VLM runner only (no LLM needed for CLIP captions)
        vlm_runner = VLLMModelRunner(model="Qwen/Qwen3-VL-30B-A3B-Thinking")
        
        # Load the paper data (can be filtered or scored)
        with open(input_file_path, 'r') as f:
            paper_data = json.load(f)
        
        # Get original image count
        original_count = len(paper_data.get("images", []))
        
        if original_count == 0:
            return {
                "status": "success",
                "original_file": input_file_path,
                "output_file": None,
                "image_count": 0,
                "clip_pairs": []
            }
        
        # Create all caption generation tasks
        image_tasks = []
        for i, image in enumerate(paper_data.get("images", [])):
            image_path = image.get("full_image_path", "")
            task = generate_clip_caption(
                vlm_runner,
                image.get("description", ""),
                image.get("context_paragraphs", []),
                image_path
            )
            image_tasks.append((i, task))
        
        # Run all caption generation tasks in parallel
        tasks = [task for _, task in image_tasks]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and build CLIP pairs
        clip_pairs = []
        enriched_paper_data = paper_data.copy()
        
        for (i, _), result in zip(image_tasks, all_results):
            if isinstance(result, Exception):
                result = {
                    "status": "error",
                    "error": str(result),
                    "clip_caption": "",
                    "original_description": paper_data["images"][i].get("description", ""),
                    "image_path": paper_data["images"][i].get("full_image_path", ""),
                    "used_image": False
                }
            
            # Update paper data with CLIP caption
            enriched_paper_data["images"][i]["clip_caption"] = result["clip_caption"]
            enriched_paper_data["images"][i]["caption_status"] = result["status"]
            if result["status"] == "error":
                enriched_paper_data["images"][i]["caption_error"] = result.get("error", "")
            
            # Add to CLIP pairs if successful
            if result["status"] == "success" and result["clip_caption"]:
                clip_pairs.append({
                    "image_path": result["image_path"],
                    "caption": result["clip_caption"],
                    "original_description": result["original_description"]
                })
        
        # Add metadata
        enriched_paper_data["clip_metadata"] = {
            "total_images": original_count,
            "successful_captions": len(clip_pairs),
            "failed_captions": original_count - len(clip_pairs),
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "model_used": "Qwen/Qwen3-VL-30B-A3B-Thinking"
        }
        
        # Generate output file path if not provided
        if output_file_path is None:
            # Handle different input file naming patterns
            if "_scored.json" in input_file_path:
                base_path = input_file_path.rsplit('_scored.json', 1)[0]
            elif "_filtered.json" in input_file_path:
                base_path = input_file_path.rsplit('_filtered.json', 1)[0]
            elif "_extracted.json" in input_file_path:
                base_path = input_file_path.rsplit('_extracted.json', 1)[0]
            else:
                base_path = input_file_path.rsplit('.json', 1)[0]
            output_file_path = f"{base_path}_clip.json"
        
        # Save enriched data with CLIP captions
        with open(output_file_path, 'w') as f:
            json.dump(enriched_paper_data, f, indent=2)
        
        return {
            "status": "success",
            "original_file": input_file_path,
            "output_file": output_file_path,
            "image_count": original_count,
            "successful_captions": len(clip_pairs),
            "clip_pairs": clip_pairs
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "original_file": input_file_path,
            "output_file": None,
            "image_count": 0,
            "clip_pairs": []
        }

async def generate_clip_captions_parallel(
    paper_folder_path: str,
    max_concurrent_papers: int = 5,
    input_pattern: str = "*_scored.json"
) -> List[Dict[str, Any]]:
    """
    Generate CLIP captions for multiple papers in parallel.
    
    Args:
        paper_folder_path: Path to folder containing JSON files
        max_concurrent_papers: Maximum number of papers to process concurrently
        input_pattern: Glob pattern to match input files (default: *_scored.json)
    """
    # Find all matching JSON files
    json_files = glob.glob(os.path.join(paper_folder_path, "**", input_pattern), recursive=True)
    
    if not json_files:
        print(f"No {input_pattern} files found in {paper_folder_path}")
        return []
    
    print(f"Found {len(json_files)} files to process")
    
    # Process papers in parallel with semaphore control
    semaphore = asyncio.Semaphore(max_concurrent_papers)
    pbar = tqdm(total=len(json_files), desc="Papers", unit="paper")
    
    async def process_single_paper_with_logging(json_file):
        async with semaphore:
            try:
                result = await generate_clip_captions_for_paper(json_file)
                pbar.update(1)
                return result
            except Exception as e:
                pbar.update(1)
                return {
                    "status": "error",
                    "error": str(e),
                    "original_file": json_file,
                    "output_file": None
                }
    
    # Create tasks for all papers
    tasks = [process_single_paper_with_logging(json_file) for json_file in json_files]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    pbar.close()
    
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
                "output_file": None
            })
    
    return processed_results

async def generate_clip_captions_batches(
    base_path: str = "/bigdisk/minhpvt/quick-test/na/ML/data/batches",
    batch_start: int = 1,
    batch_end: int = 200,
    max_concurrent_papers: int = 5,
    input_pattern: str = "*_scored.json"
) -> Dict[str, Any]:
    """
    Generate CLIP captions for multiple batches of papers.
    
    Args:
        base_path: Base path to the batches folder
        batch_start: Starting batch number (inclusive)
        batch_end: Ending batch number (inclusive)
        max_concurrent_papers: Maximum number of papers to process concurrently per batch
        input_pattern: Glob pattern to match input files
    """
    total_papers_processed = 0
    total_papers_success = 0
    total_papers_failed = 0
    total_images = 0
    total_captions = 0
    batch_results = {}
    all_clip_pairs = []
    
    print(f"Starting CLIP caption generation from batch {batch_start} to {batch_end}")
    print(f"Max concurrent papers per batch: {max_concurrent_papers}")
    print(f"Input pattern: {input_pattern}")
    
    # Progress bar for batches
    batch_pbar = tqdm(range(batch_start, batch_end + 1), desc="Batches", unit="batch", position=0)
    
    for batch_num in batch_pbar:
        batch_folder = f"batch_{batch_num:04d}"
        batch_path = os.path.join(base_path, batch_folder)
        papers_path = os.path.join(batch_path, "papers")
        
        batch_pbar.set_description(f"Batch {batch_folder}")
        
        # Check if batch folder exists
        if not os.path.exists(papers_path):
            batch_results[batch_folder] = {
                "status": "folder_not_found",
                "papers_processed": 0,
                "papers_success": 0,
                "papers_failed": 0
            }
            continue
        
        try:
            batch_start_time = asyncio.get_event_loop().time()
            
            # Generate captions for papers in this batch
            results = await generate_clip_captions_parallel(
                papers_path,
                max_concurrent_papers,
                input_pattern
            )
            
            # Process batch results
            batch_processed = len(results)
            batch_success = sum(1 for r in results if r["status"] == "success")
            batch_failed = sum(1 for r in results if r["status"] == "error")
            batch_images = sum(r.get("image_count", 0) for r in results)
            batch_captions = sum(r.get("successful_captions", 0) for r in results)
            
            # Collect CLIP pairs
            for r in results:
                if r.get("clip_pairs"):
                    all_clip_pairs.extend(r["clip_pairs"])
            
            batch_end_time = asyncio.get_event_loop().time()
            batch_duration = batch_end_time - batch_start_time
            
            total_papers_processed += batch_processed
            total_papers_success += batch_success
            total_papers_failed += batch_failed
            total_images += batch_images
            total_captions += batch_captions
            
            batch_results[batch_folder] = {
                "status": "completed",
                "papers_processed": batch_processed,
                "papers_success": batch_success,
                "papers_failed": batch_failed,
                "images_processed": batch_images,
                "captions_generated": batch_captions,
                "duration_seconds": batch_duration
            }
            
            # Update batch progress bar with stats
            batch_pbar.set_postfix({
                "success": batch_success,
                "captions": batch_captions,
                "time": f"{batch_duration:.1f}s"
            })
            
        except Exception as e:
            batch_results[batch_folder] = {
                "status": "batch_failed",
                "papers_processed": 0,
                "papers_success": 0,
                "papers_failed": 0,
                "error": str(e)
            }
    
    batch_pbar.close()
    
    # Create comprehensive results
    clip_results = {
        "clip_metadata": {
            "batch_start": batch_start,
            "batch_end": batch_end,
            "max_concurrent_papers": max_concurrent_papers,
            "input_pattern": input_pattern,
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "model_used": "Qwen/Qwen3-VL-30B-A3B-Thinking"
        },
        "summary": {
            "total_papers_processed": total_papers_processed,
            "total_papers_success": total_papers_success,
            "total_papers_failed": total_papers_failed,
            "total_images": total_images,
            "total_captions_generated": total_captions,
            "caption_success_rate": total_captions / total_images * 100 if total_images > 0 else 0
        },
        "batch_details": batch_results
    }
    
    # Save results
    results_file = os.path.join(
        base_path,
        f"clip_caption_results_{batch_start}_{batch_end}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(results_file, 'w') as f:
        json.dump(clip_results, f, indent=2)
    
    # Save all CLIP pairs to a separate file for easy training data access
    clip_pairs_file = os.path.join(
        base_path,
        f"clip_pairs_{batch_start}_{batch_end}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(clip_pairs_file, 'w') as f:
        json.dump({
            "metadata": {
                "total_pairs": len(all_clip_pairs),
                "batch_range": f"{batch_start}-{batch_end}",
                "generation_timestamp": datetime.datetime.now().isoformat()
            },
            "clip_pairs": all_clip_pairs
        }, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"CLIP CAPTION GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Batches processed: {batch_end - batch_start + 1}")
    print(f"Total papers processed: {total_papers_processed}")
    print(f"Papers success: {total_papers_success}")
    print(f"Papers failed: {total_papers_failed}")
    print(f"Total images: {total_images}")
    print(f"Captions generated: {total_captions} ({total_captions / total_images * 100:.1f}%)" if total_images > 0 else "Captions generated: 0")
    print(f"Total CLIP pairs: {len(all_clip_pairs)}")
    print(f"\nResults saved to: {results_file}")
    print(f"CLIP pairs saved to: {clip_pairs_file}")
    
    return clip_results

if __name__ == "__main__":
    # Example usage
    loop = asyncio.get_event_loop()
    
    # Generate CLIP captions for a single paper
    # result = loop.run_until_complete(
    #     generate_clip_captions_for_paper(
    #         "/path/to/paper_scored.json"
    #     )
    # )
    # print(result)
    
    # Generate CLIP captions for batches
    results = loop.run_until_complete(
        generate_clip_captions_batches(
            batch_start=1,
            batch_end=1,
            max_concurrent_papers=100,
            input_pattern="*_filtered.json" 
        )
    )
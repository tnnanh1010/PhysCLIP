import datetime
import os
import json
import asyncio
import warnings
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Union
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import sys
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from vllm_server.vllm_chat_completion import VLLMModelRunner
import re
PROMPT_TEMPLATE ="""
Please assign a relevance score from 0 to 5 based on the following criteria:

Scoring Criteria:
	•	5: The image is entirely about astrophysics-related sciences, such as astronomy, cosmology, stellar physics, planetary science, galactic dynamics, or space exploration. It contains domain-specific terms, technical discussions, or observational/theoretical methods from these fields.
	•	4: The image is primarily about astrophysics-related sciences but includes some secondary content from other disciplines (e.g., engineering, computer science, or general physics) that support or relate to astrophysical research.
	•	3: The image contains significant astrophysical content, but it's not the central focus. It may discuss astrophysics in the context of interdisciplinary applications or general science.
	•	2: The image includes minor astrophysical references, such as mentioning a celestial object, astronomical phenomenon, or astrophysical term without in-depth discussion.
	•	1: The image has very little astrophysical relevance, perhaps referencing an astronomical term or concept in passing.
	•	0: The image has no relevance to astrophysics, astronomy, or related fields.

If the image contain tables
The image is given as:
{document}

After examining the image:
	•	Briefly justify your total score (up to 100 words).
	•	Conclude with the score using the format: Score: <total points>
"""

async def process_paper_components(inputs):
    """
    Process image components of a paper and return them formatted for scoring.
    
    Returns:
        dict: Contains formatted content for images only (with image paths)
    """
    
    # Process images only
    image_contents = []
    for image in inputs["images"]:
        content = image["description"] + "\n" + "\n".join(image["context_paragraphs"])
        image_contents.append({
            "formatted_content": content,
            "image_path": image.get("full_image_path", "")
        })
    
    return {
        "images": image_contents
    }

async def score_component(model_runner, content, component_type, image_path=None):
    """
    Score a single component using the model (VLM for images).
    
    Args:
        model_runner: The model runner instance
        content: The formatted content to score
        component_type: Type of component (image)
        image_path: Path to the image file (for VLM scoring)
    
    Returns:
        dict: Contains score and explanation
    """
    import os
    
    # Check if we can use VLM with the image
    use_vlm = image_path and os.path.exists(image_path)
    
    if use_vlm:
        # Use VLM with image
        image_url = f"file://{os.path.abspath(image_path)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are a domain expert evaluating the relevance of a image to astrophysics and astronomy sciences.\n\n{PROMPT_TEMPLATE.format(document=content)}"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        model_type = "vlm"
    else:
        # Fallback to text-only LLM
        messages = [
            {"role": "system", "content": "You are a domain expert evaluating the relevance of a document to astrophysics and astronomy sciences."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(document=content)}
        ]
        model_type = "llm"
    
    try:
        response = await model_runner.run_model_completion(messages, model_type)
        result_text = model_runner.extract_message_content(response)
        
        # Extract score from the response
        import re
        score_match = re.search(r'Score:\s*(\d+)', result_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 0
        
        # Extract explanation (everything before "Score:")
        explanation_match = re.split(r'Score:\s*\d+', result_text, flags=re.IGNORECASE)
        explanation = explanation_match[0].strip() if explanation_match else result_text.strip()
        
        return {
            "score": score,
            "explanation": explanation,
            "full_response": result_text
        }
    except Exception as e:
        print(f"Error scoring {component_type}: {str(e)}")
        return {
            "score": 0,
            "explanation": f"Error occurred during scoring: {str(e)}",
            "full_response": ""
        }

async def score_component_batch(model_runner, content_batch, component_types, image_paths=None):
    """
    Score a batch of components in parallel.
    
    Args:
        model_runner: The model runner instance
        content_batch: List of formatted content to score
        component_types: List of component types corresponding to content
        image_paths: List of image paths for VLM scoring (optional)
    
    Returns:
        list: List of scoring results
    """
    if image_paths is None:
        image_paths = [None] * len(content_batch)
    
    tasks = []
    for content, component_type, image_path in zip(content_batch, component_types, image_paths):
        task = score_component(model_runner, content, component_type, image_path)
        tasks.append(task)
    
    # Run all scoring tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error scoring {component_types[i]}: {str(result)}")
            processed_results.append({
                "score": 0,
                "explanation": f"Error occurred during scoring: {str(result)}",
                "full_response": ""
            })
        else:
            processed_results.append(result)
    
    return processed_results

async def process_and_score_paper_parallel(input_file_path, output_file_path=None, max_concurrent=10):
    """
    Process and score all components of a paper in parallel using VLM.
    
    Args:
        input_file_path: Path to the extracted JSON file
        output_file_path: Path to save the scored results (optional)
        max_concurrent: Maximum number of concurrent API calls
    
    Returns:
        dict: The processed paper with scores
    """
    # Use VLM model for image scoring
    model_runner = VLLMModelRunner(model="Qwen/Qwen3-VL-30B-A3B-Instruct")
    
    # Load the extracted paper data
    with open(input_file_path, 'r') as f:
        paper_data = json.load(f)
    
    # Process components
    components = await process_paper_components(paper_data)
    
    # Prepare all content for parallel processing
    all_content = []
    all_types = []
    all_image_paths = []
    component_indices = []
    
    # Collect images only
    for i, image_data in enumerate(components["images"]):
        all_content.append(image_data["formatted_content"])
        all_types.append("image")
        all_image_paths.append(image_data.get("image_path", ""))
        component_indices.append(("image", i))
    
    # print(f"Scoring {len(all_content)} image components with VLM in parallel...")
    
    # Process in batches to avoid overwhelming the servers
    batch_size = max_concurrent
    all_results = []
    
    for i in range(0, len(all_content), batch_size):
        batch_content = all_content[i:i + batch_size]
        batch_types = all_types[i:i + batch_size]
        batch_image_paths = all_image_paths[i:i + batch_size]
        
        # print(f"Processing batch {i//batch_size + 1}/{(len(all_content) + batch_size - 1)//batch_size}")
        
        batch_results = await score_component_batch(model_runner, batch_content, batch_types, batch_image_paths)
        all_results.extend(batch_results)
    
    # Assign results back to the original data structure
    for (component_type, index), result in zip(component_indices, all_results):
        if component_type == "image":
            paper_data["images"][index]["astro_relevance_score"] = result["score"]
            paper_data["images"][index]["astro_relevance_explanation"] = result["explanation"]
            paper_data["images"][index]["astro_relevance_full_response"] = result["full_response"]
    
    # Calculate overall paper score (images only)
    all_scores = []
    for image in paper_data["images"]:
        all_scores.append(image["astro_relevance_score"])
    
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
    paper_data["overall_astro_relevance_score"] = overall_score
    paper_data["component_count"] = {
        "images": len(paper_data["images"]),
        "total_components": len(all_scores)
    }
    
    # Save the results
    if output_file_path is None:
        base_path = input_file_path.rsplit('.', 1)[0]
        output_file_path = f"{base_path}_scored.json"
    
    with open(output_file_path, 'w') as f:
        json.dump(paper_data, f, indent=2)
    
    # print(f"Scored paper saved to: {output_file_path}")
    # print(f"Overall astro-relevance score: {overall_score:.2f}")
    
    return paper_data

async def process_multiple_papers_parallel(paper_folder_path, max_concurrent_papers=4, max_concurrent_components=10):
    """
    Process multiple papers in parallel.
    
    Args:
        paper_folder_path: Path to folder containing extracted JSON files
        max_concurrent_papers: Maximum number of papers to process concurrently
        max_concurrent_components: Maximum number of components to score concurrently per paper
    """
    import glob
    
    json_files = glob.glob(os.path.join(paper_folder_path, "*_extracted.json"))
    
    # Process papers in batches
    semaphore = asyncio.Semaphore(max_concurrent_papers)
    
    async def process_single_paper(json_file):
        async with semaphore:
            print(f"\nProcessing: {json_file}")
            try:
                return await process_and_score_paper_parallel(json_file, max_concurrent=max_concurrent_components)
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                return None
    
    # Create tasks for all papers
    tasks = [process_single_paper(json_file) for json_file in json_files]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Report results
    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    print(f"\nCompleted processing {successful}/{len(json_files)} papers successfully")
    
    return results

async def process_multiple_batches_parallel(
    base_path="/bigdisk/minhpvt/quick-test/na/ML/data/batches",
    batch_start=1,
    batch_end=200,
    max_concurrent_papers=4,
    max_concurrent_components=10,
    resume_from_batch=None
):
    """
    Process multiple batches of papers in parallel.
    
    Args:
        base_path: Base path to the batches folder
        batch_start: Starting batch number (inclusive)
        batch_end: Ending batch number (inclusive)
        max_concurrent_papers: Maximum number of papers to process concurrently per batch
        max_concurrent_components: Maximum number of components to score concurrently per paper
        resume_from_batch: Resume processing from a specific batch number (optional)
    """
    import glob
    from pathlib import Path
    
    total_papers_processed = 0
    total_papers_failed = 0
    batch_results = {}
    
    # Determine actual start batch
    actual_start = resume_from_batch if resume_from_batch else batch_start
    
    print(f"Starting batch processing from batch {actual_start} to {batch_end}")
    print(f"Max concurrent papers per batch: {max_concurrent_papers}")
    print(f"Max concurrent components per paper: {max_concurrent_components}")
    
    # Progress bar for batches
    batch_pbar = tqdm(range(actual_start, batch_end + 1), desc="Batches", unit="batch", position=0)
    
    for batch_num in batch_pbar:
        batch_folder = f"batch_{batch_num:04d}"  # Format as batch_0001, batch_0002, etc.
        batch_path = os.path.join(base_path, batch_folder)
        papers_path = os.path.join(batch_path, "papers")
        
        batch_pbar.set_description(f"Batch {batch_folder}")
        
        # Check if batch folder exists
        if not os.path.exists(papers_path):
            print(f"Papers folder not found: {papers_path}")
            batch_results[batch_folder] = {"status": "folder_not_found", "papers_processed": 0, "papers_failed": 0}
            continue
        
        # Find all paper folders in this batch
        paper_folders = [d for d in os.listdir(papers_path) if os.path.isdir(os.path.join(papers_path, d))]
        
        if not paper_folders:
            print(f"No paper folders found in {papers_path}")
            batch_results[batch_folder] = {"status": "no_papers", "papers_processed": 0, "papers_failed": 0}
            continue
        
        print(f"Found {len(paper_folders)} paper folders in {batch_folder}")
        
        # Collect all extracted JSON files in this batch
        json_files = []
        for paper_folder in paper_folders:
            paper_folder_path = os.path.join(papers_path, paper_folder)
            # Look for extracted JSON files in each paper folder
            extracted_files = glob.glob(os.path.join(paper_folder_path, "*_extracted.json"))
            json_files.extend(extracted_files)
        
        if not json_files:
            print(f"No extracted JSON files found in {batch_folder}")
            batch_results[batch_folder] = {"status": "no_extracted_files", "papers_processed": 0, "papers_failed": 0}
            continue
        
        print(f"Found {len(json_files)} extracted JSON files to process")
        
        # Process papers in this batch
        try:
            batch_start_time = asyncio.get_event_loop().time()
            
            # Process papers in parallel with semaphore control
            semaphore = asyncio.Semaphore(max_concurrent_papers)
            
            # Progress bar for papers in this batch
            paper_pbar = tqdm(total=len(json_files), desc=f"  Papers", unit="paper", position=1, leave=False)
            
            async def process_single_paper_with_logging(json_file):
                async with semaphore:
                    paper_name = os.path.basename(json_file)
                    try:
                        result = await process_and_score_paper_parallel(
                            json_file, 
                            max_concurrent=max_concurrent_components
                        )
                        paper_pbar.update(1)
                        return {"file": json_file, "status": "success", "result": result}
                    except Exception as e:
                        paper_pbar.update(1)
                        return {"file": json_file, "status": "failed", "error": str(e)}
            
            # Create tasks for all papers in this batch
            tasks = [process_single_paper_with_logging(json_file) for json_file in json_files]
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Close paper progress bar
            paper_pbar.close()
            
            # Process results
            batch_successful = 0
            batch_failed = 0
            
            for result in results:
                if isinstance(result, dict):
                    if result["status"] == "success":
                        batch_successful += 1
                    else:
                        batch_failed += 1
                else:
                    batch_failed += 1
            
            batch_end_time = asyncio.get_event_loop().time()
            batch_duration = batch_end_time - batch_start_time
            
            total_papers_processed += batch_successful
            total_papers_failed += batch_failed
            
            batch_results[batch_folder] = {
                "status": "completed",
                "papers_processed": batch_successful,
                "papers_failed": batch_failed,
                "total_papers": len(json_files),
                "duration_seconds": batch_duration
            }
            
            # Update batch progress bar with stats
            batch_pbar.set_postfix({
                "success": batch_successful,
                "failed": batch_failed,
                "time": f"{batch_duration:.1f}s"
            })
            
        except Exception as e:
            print(f"Error processing batch {batch_folder}: {str(e)}")
            batch_results[batch_folder] = {
                "status": "batch_failed",
                "papers_processed": 0,
                "papers_failed": len(json_files),
                "error": str(e)
            }
            total_papers_failed += len(json_files)
    
    # Close batch progress bar
    batch_pbar.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Batches processed: {batch_end - actual_start + 1}")
    print(f"Total papers processed successfully: {total_papers_processed}")
    print(f"Total papers failed: {total_papers_failed}")
    print(f"Overall success rate: {total_papers_processed / (total_papers_processed + total_papers_failed) * 100:.2f}%" if (total_papers_processed + total_papers_failed) > 0 else "N/A")
    
    # Save detailed results
    results_file = os.path.join(base_path, f"batch_processing_results_{actual_start}_{batch_end}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "batch_start": actual_start,
                "batch_end": batch_end,
                "total_papers_processed": total_papers_processed,
                "total_papers_failed": total_papers_failed,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "batch_details": batch_results
        }, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    
    return batch_results

async def resume_batch_processing(results_file_path, max_concurrent_papers=4, max_concurrent_components=10):
    """
    Resume batch processing from a previous results file.
    
    Args:
        results_file_path: Path to the previous results JSON file
        max_concurrent_papers: Maximum concurrent papers per batch
        max_concurrent_components: Maximum concurrent components per paper
    """
    with open(results_file_path, 'r') as f:
        previous_results = json.load(f)
    
    # Find the last successfully completed batch
    last_completed_batch = 0
    for batch_name, batch_info in previous_results["batch_details"].items():
        if batch_info["status"] == "completed":
            batch_num = int(batch_name.split('_')[1])
            last_completed_batch = max(last_completed_batch, batch_num)
    
    resume_from = last_completed_batch + 1
    original_end = previous_results["summary"]["batch_end"]
    
    print(f"Resuming from batch {resume_from}")
    
    return await process_multiple_batches_parallel(
        resume_from_batch=resume_from,
        batch_end=original_end,
        max_concurrent_papers=max_concurrent_papers,
        max_concurrent_components=max_concurrent_components
    ) 
async def retry_failed_papers(
    failed_papers: List[str],
    max_concurrent_papers=4,
    max_concurrent_components=10
):
    """
    Retry processing of previously failed papers.
    
    Args:
        failed_papers: List of file paths to failed extracted JSON files
        max_concurrent_papers: Maximum concurrent papers
        max_concurrent_components: Maximum concurrent components per paper
    """
    print(f"Retrying {len(failed_papers)} failed papers...")
    
    # Process papers in parallel with semaphore control
    semaphore = asyncio.Semaphore(max_concurrent_papers)
    
    async def process_single_paper_with_logging(json_file):
        async with semaphore:
            paper_name = os.path.basename(json_file)
            try:
                result = await process_and_score_paper_parallel(
                    json_file, 
                    max_concurrent=max_concurrent_components
                )
                print(f"Successfully reprocessed: {paper_name}")
                return {"file": json_file, "status": "success", "result": result}
            except Exception as e:
                print(f"Failed again: {paper_name} with error: {str(e)}")
                return {"file": json_file, "status": "failed", "error": str(e)}
    
    # Create tasks for all failed papers
    tasks = [process_single_paper_with_logging(json_file) for json_file in failed_papers]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Report results
    successful = sum(1 for r in results if isinstance(r, dict) and r["status"] == "success")
    failed = sum(1 for r in results if isinstance(r, dict) and r["status"] == "failed")
    
    print(f"\nReprocessing completed: {successful} succeeded, {failed} failed.")
    
    return results
if __name__ == "__main__":
    # Process batches 1-200
    loop = asyncio.get_event_loop()
    
    # Option 1: Process all batches 1-200
    # results = loop.run_until_complete(
    #     process_multiple_batches_parallel(
    #         batch_start=1,
    #         batch_end=34,
    #         max_concurrent_papers=10,  # Adjust based on your server capacity
    #         max_concurrent_components=40  # Adjust based on your server capacity
    #     )
    # )

    # Option 2: Retry failed papers from a previous run
    failed_papers_list = ['/bigdisk/minhpvt/quick-test/na/ML/data/batches/batch_0034/papers/tmp00a8nl6e/tmp00a8nl6e_extracted_scored.json']
    results = loop.run_until_complete(
        retry_failed_papers(
            failed_papers=failed_papers_list,
            max_concurrent_papers=1,
            max_concurrent_components=20
        )
    )
    # Option 3: Resume from a previous run
    # results = loop.run_until_complete(
    #     resume_batch_processing(
    #         "/mnt/data/bio_pdf/processed_pdfs/2020-2025/batches/batch_processing_results_1_200_20250101_123456.json",
    #         max_concurrent_papers=6,
    #         max_concurrent_components=15
    #     )
    # )
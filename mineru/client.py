import base64
import requests
import os
from loguru import logger
import asyncio
import aiohttp
from pathlib import Path
from pypdf import PdfReader
import json
from tqdm.asyncio import tqdm
import asyncio
import shutil

async def track_progress(tasks):
    progress = tqdm(total=len(tasks))
    results = []

    async def wrapped(task):
        result = await task
        progress.update(1)
        return result

    # wrap each task
    wrapped_tasks = [wrapped(task) for task in tasks]
    results = await asyncio.gather(*wrapped_tasks)
    progress.close()
    return results

def is_valid_pdf(file_path):
    """Check if PDF file is valid and readable"""
    try:
        with open(file_path, 'rb') as f:
            PdfReader(f)
        return True
    except Exception as e:
        logger.warning(f"Invalid PDF {file_path}: {e}")
        return False


async def mineru_parse_async(session, file_path, server_url='http://127.0.0.1:8001/predict', **options):
    """
    Asynchronous version of the parse function with optimizations for large batches.
    """
    try:
        # # Check if file is valid before processing
        # if not is_valid_pdf(file_path):
        #     # Delete the file if it's invalid
        #     logger.error(f"❌ Invalid PDF file: {file_path}")
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        #     # Return error response
        #     return {
        #         'error': 'Invalid PDF',
        #         'processed_file_path': {'Error': file_path, 'Message': 'Invalid PDF format'}
        #     }
        
        # Asynchronously read and encode the file
        with open(file_path, 'rb') as f:
            file_b64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            'file': file_b64,
            'options': options
        }
        
        # Use optimized timeout for large batches
        timeout = aiohttp.ClientTimeout(total=3000, connect=30)  # 30 minute timeout

        # Use the aiohttp session to send the request
        async with session.post(server_url, json=payload, timeout = timeout) as response:
            if response.status == 404:
                logger.error(f"❌ Server not found at {server_url}. Is the MinerU server running?")
                return {
                    'error': 'Server not found',
                    'processed_file_path': {'Error': file_path, 'Message': 'Server not found'}
                }

            if response.status == 200:
                result = await response.json()
                logger.debug(f"✅ Processed: {os.path.basename(file_path)}")  # Use debug level to reduce logs
                
                # Add processed file path to metadata
                result['processed_file_path'] = {'Success': file_path, 'Output': result.get('output_dir', 'N/A')}
                return result
            else:
                error_text = await response.text()
                logger.warning(f"❌ Server error for {os.path.basename(file_path)}: {error_text[:100]}...")
                return {
                    'error': error_text,
                    'processed_file_path': {'Error': file_path, 'Message': error_text}
                }

    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout processing {os.path.basename(file_path)}")
        return {
            'error': 'Timeout',
            'processed_file_path': {'Error': file_path, 'Message': 'Processing timeout'}
        }
    except Exception as e:
        logger.error(f"❌ Failed to process {os.path.basename(file_path)}: {e}")
        return {
            'error': str(e),
            'processed_file_path': {'Error': file_path, 'Message': str(e)}
        }


async def main():
    """
    Main function to run all parsing tasks concurrently.
    """
    test_files = []
    test_files = []
    base_path = '/bigdisk/minhpvt/quick-test/na/train_test_set'
    metadata_path = '/bigdisk/minhpvt/quick-test/na/train_test_set/metadata/processed_path.json'
    output_base_path = '/bigdisk/minhpvt/quick-test/na/ML/data/batches'
    base_path = Path(base_path)

    # Create metadata directory at startup
    metadata_dir = os.path.dirname(metadata_path)
    os.makedirs(metadata_dir, exist_ok=True)
    logger.info(f"📁 Ensured metadata directory exists: {metadata_dir}")

    # Collect all already processed papers from batch folders
    already_processed = set()
    if os.path.exists(output_base_path):
        for batch_dir in Path(output_base_path).glob('batch_*'):
            papers_folder = batch_dir / 'papers'
            if papers_folder.exists():
                for paper_dir in papers_folder.iterdir():
                    if paper_dir.is_dir():
                        already_processed.add(paper_dir.name)
    
    logger.info(f"📋 Found {len(already_processed)} already processed papers")

    for pdf_file in base_path.glob("*.pdf"):
        file_name = pdf_file.stem
        # Skip if already processed
        if file_name in already_processed:
            continue
        test_files.append(f'{base_path}/{file_name}.pdf')
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    if not existing_files:
        logger.warning("No new files to process (all files already extracted or no files found).")
        return
    logger.info(f"Found {len(existing_files)} new files to process (skipped {len(already_processed)} already processed)...")
    # Create an aiohttp session with optimized settings for massive throughput
    connector = aiohttp.TCPConnector(
        limit=1000,
        limit_per_host=500,
        ttl_dns_cache=300,
        use_dns_cache=True,
        enable_cleanup_closed=True
    )
    
    # Session timeout should be longer than individual request timeouts
    timeout = aiohttp.ClientTimeout(
        total=None,    # No total session timeout
        connect=30,     # 30s to connect
        sock_read=None  # No read timeout at session level
    )
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,  
        headers={'Connection': 'keep-alive'}
    ) as session:

        # === Custom Options ===
        custom_options = {
            'backend': 'pipeline', 'lang': 'en', 'method': 'auto',
            'formula_enable': False, 'table_enable': False
        }
        # 'backend': 'sglang-engine' requires 24+ GB VRAM per worker

        # Process files in chunks to avoid memory overload
        CHUNK_SIZE = 300  # Process 300 files at a time
        MAX_CONCURRENT = 100  # Max concurrent requests to server
        
        all_results = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async def limited_parse(file_path):
            async with semaphore:
                return await mineru_parse_async(session, file_path, **custom_options)
        
        # Process files in chunks
        for i in range(0, len(existing_files), CHUNK_SIZE):
            chunk = existing_files[i:i + CHUNK_SIZE]
            logger.info(f"🔄 Processing chunk {i//CHUNK_SIZE + 1}/{(len(existing_files) + CHUNK_SIZE - 1)//CHUNK_SIZE} ({len(chunk)} files)")
            
            # Create tasks for this chunk with concurrency limit
            chunk_tasks = [limited_parse(file_path) for file_path in chunk]
            
            # Process chunk with progress tracking
            chunk_results = await track_progress(chunk_tasks)
            all_results.extend(chunk_results)
            
            # Move papers to batch folder after each chunk
            batch_num = (i // CHUNK_SIZE) + 1
            batch_folder = f'/bigdisk/minhpvt/quick-test/na/ML/data/batches/batch_{batch_num:04d}'
            papers_folder = os.path.join(batch_folder, 'papers')
            metadata_folder = os.path.join(batch_folder, 'metadata')
            
            # Create batch directories
            os.makedirs(papers_folder, exist_ok=True)
            os.makedirs(metadata_folder, exist_ok=True)
            
            # Move papers from this chunk to batch folder
            for result in chunk_results:
                if 'error' not in result and 'output_dir' in result:
                    source_dir = result['output_dir']
                    if os.path.exists(source_dir):
                        paper_name = os.path.basename(source_dir)
                        target_dir = os.path.join(papers_folder, paper_name)
                        try:
                            shutil.move(source_dir, target_dir)
                            # Update the result to point to new location
                            result['output_dir'] = target_dir
                            result['processed_file_path']['Output'] = target_dir
                        except Exception as e:
                            logger.warning(f"Failed to move {source_dir}: {e}")
            
            # Save batch metadata
            batch_metadata = [result['processed_file_path'] for result in chunk_results]
            metadata_file = os.path.join(metadata_folder, f'batch_{batch_num:04d}_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(batch_metadata, f, indent=4)
            
            logger.info(f"✅ Completed chunk {i//CHUNK_SIZE + 1}. Total processed: {len(all_results)}. Moved {len([r for r in chunk_results if 'error' not in r])} papers to batch_{batch_num:04d}")

        logger.info(f"All Results: {len(all_results)} total processed")
        
        # Also save the traditional metadata file for backward compatibility
        processed_file_paths = [result['processed_file_path'] for result in all_results]
        # Makedir if not exist
        if not os.path.exists(os.path.dirname(metadata_path)):
            os.makedirs(os.path.dirname(metadata_path))
        with open(metadata_path, 'w') as f:
            json.dump(processed_file_paths, f, indent=4)
        logger.info(f"✅✅✅Processed file paths saved to {metadata_path}")
    logger.info("🎉 All processing completed!")

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())
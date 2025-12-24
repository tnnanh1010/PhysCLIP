import os
import base64
import tempfile
from pathlib import Path
import litserve as ls
from fastapi import HTTPException
from loguru import logger

from mineru.cli.common import do_parse, read_fn
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram
from _config_endpoint import config_endpoint
import torch
import gc

import os
import shutil
from pathlib import Path

def simplify_paper_structure(paper_folder):
    """
    Simplify the paper folder structure by:
    1. Moving from base_dir/paper_name/paper_name/auto/* to base_dir/paper_name/*
    2. Keeping only the images folder and .md file from the auto folder
    """
    paper_folder = Path(paper_folder)
    
    # Iterate through all paper folders in base directory
    if paper_folder.is_dir():
        print(f"Processing paper folder: {paper_folder.name}")
        
        # Look for the nested folder structure (paper_name/paper_name/auto)
        nested_paper_folder = paper_folder / paper_folder.name
        if nested_paper_folder.exists():
            auto_folder = nested_paper_folder / "auto"
            if auto_folder.exists():
                print(f"  Found auto folder: {auto_folder}")
                
                # Find the .md file and images folder
                images_folder = auto_folder / "images"
                md_files = list(auto_folder.glob("*.md"))
                
                # Create new structure directly under paper folder
                if images_folder.exists():
                    new_images_path = paper_folder / "images"
                    if new_images_path.exists():
                        shutil.rmtree(new_images_path)
                    shutil.move(str(images_folder), str(new_images_path))
                    print(f"  Moved images folder to: {new_images_path}")
                
                # Move .md file(s)
                for md_file in md_files:
                    new_md_path = paper_folder / md_file.name
                    if new_md_path.exists():
                        new_md_path.unlink()
                    shutil.move(str(md_file), str(new_md_path))
                    print(f"  Moved {md_file.name} to: {new_md_path}")
                
                # Remove the nested folder structure
                try:
                    shutil.rmtree(nested_paper_folder)
                    print(f"  Removed nested folder: {nested_paper_folder}")
                except Exception as e:
                    print(f"  Error removing nested folder: {e}")
            else:
                print(f"  No auto folder found in {nested_paper_folder}")
        else:
            print(f"  No nested folder found for {paper_folder.name}")


class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir='/tmp'):
        super().__init__()
        self.output_dir = output_dir

    def setup(self, device):
        """Setup environment variables exactly like MinerU CLI does"""
        logger.info(f"Setting up on device: {device}")
                
        if os.getenv('MINERU_DEVICE_MODE', None) == None:
            os.environ['MINERU_DEVICE_MODE'] = device if device != 'auto' else get_device()

        device_mode = os.environ['MINERU_DEVICE_MODE']
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE', None) == None:
            if device_mode.startswith("cuda") or device_mode.startswith("npu"):
                vram = round(get_vram(device_mode))
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = str(vram)
            else:
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = '1'
        logger.info(f"MINERU_VIRTUAL_VRAM_SIZE: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}")

        if os.getenv('MINERU_MODEL_SOURCE', None) in ['huggingface', None]:
            config_endpoint()
        logger.info(f"MINERU_MODEL_SOURCE: {os.environ['MINERU_MODEL_SOURCE']}")


    def decode_request(self, request):
        """Decode file and options from request"""
        try:
            file_b64 = request['file']
            options = request.get('options', {})
            
            if not file_b64:
                raise ValueError("No file data provided")
                
            file_bytes = base64.b64decode(file_b64)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(file_bytes)
                temp_file = Path(temp.name)
                
        except (KeyError, ValueError, base64.binascii.Error) as e:
            logger.error(f"Invalid request format: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
            
        return {
            'input_path': str(temp_file),
            'backend': options.get('backend', 'pipeline'),
            'method': options.get('method', 'auto'),
            'lang': options.get('lang', 'en'),
            'formula_enable': options.get('formula_enable', False),
            'table_enable': options.get('table_enable', False),
            'start_page_id': options.get('start_page_id', 0),
            'end_page_id': options.get('end_page_id', None),
            'server_url': options.get('server_url', None),
        }

    def predict(self, inputs):
        """Call MinerU's do_parse - same as CLI"""
        input_path = inputs['input_path']
        output_dir = Path(self.output_dir) / Path(input_path).stem
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            file_name = Path(input_path).stem
            pdf_bytes = read_fn(Path(input_path))
            
            do_parse(
                output_dir=str(output_dir),
                pdf_file_names=[file_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[inputs['lang']],
                backend=inputs['backend'],
                parse_method=inputs['method'],
                formula_enable=inputs['formula_enable'],
                table_enable=inputs['table_enable'],
                server_url=inputs['server_url'],
                start_page_id=inputs['start_page_id'],
                end_page_id=inputs['end_page_id']
            )
            
             # Clear GPU cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()

            # Simplify the folder structure after parsing
            try:
                simplify_paper_structure(output_dir)
                logger.info(f"✅ Simplified structure for: {output_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to simplify structure for {output_dir}: {e}")
                # Don't fail the whole request if structure simplification fails

            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Cleanup temp file
            if Path(input_path).exists():
                Path(input_path).unlink()

    def encode_response(self, response):
        return {'output_dir': response}

if __name__ == '__main__':
    server = ls.LitServer(
        MinerUAPI(output_dir='/bigdisk/minhpvt/quick-test/na/ML/data'),
        accelerator='auto',
        devices=[0],
        workers_per_device=6,  # Increased workers for higher throughput
        timeout=False,
        max_batch_size=1,       # Process one PDF at a time for stability
        # batch_timeout=0.1       # Quick batching timeout
    )
    logger.info("Starting MinerU server on port 8001 with high-throughput config")
    server.run(port=8001, generate_client_file=False) 
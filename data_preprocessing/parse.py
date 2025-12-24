import re
import json
import os
from pathlib import Path

class Parser():
    def extract_paper_content(self, md_file_path, paper_folder_path):
        """
        Extract paragraphs, images, and tables information from a markdown paper
        
        Args:
            md_file_path: Path to the markdown file
            paper_folder_path: Path to the paper folder containing images
        
        Returns:
            dict: Contains paragraphs, images, and tables data
        """
        
        with open(md_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
                
        # Extract image information
        images = self.extract_image_information(content, paper_folder_path)
        
        
        return {
            "images": images,
        }

    def extract_image_information(self, content, paper_folder_path):
        """Extract image paths, descriptions, and context paragraphs"""
        
        images = []
        lines = content.split('\n')
        
        # Pattern to match images: ![alt](path)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for i, line in enumerate(lines):
            image_matches = re.findall(image_pattern, line)
            
            for _, image_path in image_matches:
                # Get full image path
                full_image_path = os.path.join(paper_folder_path, image_path)
                
                # Extract figure description (usually the next line or nearby)
                description = self.extract_figure_description(lines, i)
                
                # Extract context paragraphs that mention this figure
                context_paragraphs = self.extract_figure_context(content, description)
                
                images.append({
                    "image_path": image_path,
                    "full_image_path": full_image_path,
                    "description": description,
                    "context_paragraphs": context_paragraphs
                })
        
        return images

    def extract_figure_description(self, lines, image_line_index):
        """Extract figure description that usually appears right after the image"""
        
        description = ""
        
        # Check the next few lines for figure description
        for i in range(image_line_index + 1, min(image_line_index + 5, len(lines))):
            line = lines[i].strip()
            
            # Look for "Fig. X" or "Figure X" pattern (but not "Table")
            # Updated to handle cases like "Figure 5Bmicrobiota" and "Figures"
            if re.match(r'^(Fig\.|Figures?)\s*\d+[A-Z]*', line, re.IGNORECASE) and not re.match(r'^Table', line, re.IGNORECASE):
                description = line
                break
            # Sometimes description might be on the same line or next line without Fig prefix
            elif line and not line.startswith('#') and not line.startswith('!') and not re.match(r'^Table', line, re.IGNORECASE):
                # Check if this looks like a description
                if len(line) > 20:  # Assume descriptions are reasonably long
                    description = line
                    break
        
        return description
    
    def extract_figure_context(self, content, description, chunk_size=3):
        """Extract paragraphs that mention the figure, extend to chunk_size paragraphs after"""
        
        context_paragraphs = []
        
        # Extract figure number from description
        # Updated to handle cases like "Figure 5Bmicrobiota" and "Figures"
        fig_number_match = re.search(r'(Fig\.|Figures?)\s*(\d+)[A-Z]*', description, re.IGNORECASE)
        if not fig_number_match:
            return context_paragraphs
        
        fig_number = fig_number_match.group(2)
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')

        # Look for paragraphs and the next chunk_size-1 paragraphs that mention this figure
        for i in range(len(paragraphs) - chunk_size + 1):
            # Look for references like "Figure 1", "Fig. 1", "Figures 4A, D", "(Figure 1)", etc.
            # Updated pattern to handle subfigures and "Figures" plural
            paragraph = paragraphs[i]
            if re.search(rf'\b(Fig\.|Figures?)\s*{fig_number}[A-Z,\s]*\b', paragraph, re.IGNORECASE):
                # Clean up the paragraph
                clean_paragraph = re.sub(r'\n+', ' ', paragraph).strip()
                if clean_paragraph in context_paragraphs:
                    continue
                if clean_paragraph and len(clean_paragraph) > 50 and not clean_paragraph.startswith('![]'):  # Filter out very short references and image descriptions that already extracted
                    context_paragraphs.append(clean_paragraph)
                # Append the next chunk_size-1 paragraphs
                for j in range(1, chunk_size):
                    if i + j < len(paragraphs):
                        next_paragraph = paragraphs[i + j]
                        clean_next_paragraph = re.sub(r'\n+', ' ', next_paragraph).strip()
                        if clean_next_paragraph and len(clean_next_paragraph) > 50 and not clean_next_paragraph.startswith('![]'):
                            context_paragraphs.append(clean_next_paragraph)

        return context_paragraphs


    def process_paper(self, paper_folder_path):
        """Process a single paper folder"""
        
        paper_path = Path(paper_folder_path)
        
        # Find the markdown file
        md_files = list(paper_path.glob("*.md"))
        if not md_files:
            print(f"No markdown file found in {paper_folder_path}")
            return
        
        md_file = md_files[0]  # Take the first .md file
        
        print(f"Processing {md_file.name}...")
        
        # Extract content
        extracted_data = self.extract_paper_content(md_file, paper_folder_path)
        
        # Save to JSON
        output_file = paper_path / f"{md_file.stem}_extracted.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted data saved to {output_file}")
        
        # Print summary
        print(f"Extracted {len(extracted_data['images'])} images")
        
        return extracted_data

    def process_all_papers_by_folder(self, base_directory):
        """Process all papers in the base directory"""
        
        base_path = Path(base_directory)
        
        for paper_folder in base_path.iterdir():
            if paper_folder.is_dir():
                print(f"\n=== Processing {paper_folder.name} ===")
                try:
                    self.process_paper(paper_folder)
                except Exception as e:
                    print(f"Error processing {paper_folder.name}: {e}")
    
    def process_all_papers_by_batches(self, base_directory, batch_start=1, batch_end=None):
        """Process all papers in batches from the base directory"""
        
        base_path = Path(base_directory)
        paper_folders = [f for f in base_path.iterdir() if f.is_dir()]
        
        # Sort folders by name to ensure consistent order
        paper_folders.sort(key=lambda x: x.name)
        
        # Process only the specified batch range
        for i, paper_folder in enumerate(paper_folders):
            batch_number = i + 1
            
            if batch_end is not None and (batch_number < batch_start or batch_number > batch_end):
                continue
            
            print(f"\n=== Processing Batch {batch_number}: {paper_folder.name} ===")
            try:
                self.process_all_papers_by_folder(paper_folder / "papers")
            except Exception as e:
                print(f"Error processing {paper_folder.name}: {e}")

    def parallel_process_all_papers_by_batches(self, base_directory, batch_start=1, batch_end=None, workers=5):
        """Process all papers in batches with concurrency limit"""
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        base_path = Path(base_directory)
        paper_folders = [f for f in base_path.iterdir() if f.is_dir()]
        
        # Sort folders by name to ensure consistent order
        paper_folders.sort(key=lambda x: x.name)
        
        # Filter folders based on batch range
        folders_to_process = []
        for i, paper_folder in enumerate(paper_folders):
            batch_number = i + 1
            
            if batch_end is not None and (batch_number < batch_start or batch_number > batch_end):
                continue
                
            folders_to_process.append((batch_number, paper_folder))
        
        def process_single_batch(batch_info):
            """Process a single batch folder"""
            batch_number, paper_folder = batch_info
            print(f"\n=== Processing Batch {batch_number}: {paper_folder.name} ===")
            try:
                self.process_all_papers_by_folder(paper_folder / "papers")
                return f"Successfully processed batch {batch_number}: {paper_folder.name}"
            except Exception as e:
                error_msg = f"Error processing {paper_folder.name}: {e}"
                print(error_msg)
                return error_msg
        
        # Process folders in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(process_single_batch, folder_info): folder_info 
                for folder_info in folders_to_process
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_batch):
                batch_info = future_to_batch[future]
                try:
                    result = future.result()
                    print(f"Completed: {result}")
                except Exception as exc:
                    batch_number, paper_folder = batch_info
                    print(f"Batch {batch_number} ({paper_folder.name}) generated an exception: {exc}")
        
        print(f"\nFinished processing {len(folders_to_process)} batches with {workers} workers")
# Usage
if __name__ == "__main__":
    # Process a single paper
    
    # paper_folder = "/Users/nhatanh10102005/Documents/Dev/paper_processed/paperrr"
    # parser = Parser()
    # extracted_data = parser.process_paper(paper_folder)
    
    # Or process all papers
    # base_directory = "/Users/nhatanh10102005/Documents/Dev/paper_processed"
    parser = Parser()
    parser.parallel_process_all_papers_by_batches('/bigdisk/minhpvt/quick-test/na/ML/data/batches', 1, 34, 200)

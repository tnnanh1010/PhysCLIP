"""
CLIP SpaceNet Benchmark - Optimized Version

Key optimizations:
1. Pre-load all images into memory (12,824 images ~= 3-4GB)
2. Batch process all epochs on each GPU
3. Maximize GPU utilization with larger batches
4. Minimize data loading overhead

Usage:
    python benchmark_spacenet_optimized.py
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.multiprocessing as mp

from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from transformers import (
    VisionTextDualEncoderModel,
    AutoTokenizer,
    CLIPImageProcessor,
)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class EvaluationResult:
    """Single evaluation result for one checkpoint."""
    epoch: int
    model_name: str
    accuracy: float
    inference_time: float
    num_samples: int
    timestamp: str


@dataclass
class BenchmarkMetadata:
    """Benchmark execution metadata."""
    dataset_path: str
    dataset_size: int
    class_distribution: Dict[str, int]
    model_paths: Dict[str, str]
    batch_size: int
    num_workers: int
    gpu_devices: List[int]
    timestamp: str
    total_evaluation_time: float


# ============================================================================
# Class Descriptions
# ============================================================================

CLASS_DESCRIPTIONS = {
    "asteroid": "A photograph of an asteroid",
    "black hole": "A photograph of a black hole",
    "comet": "A photograph of a comet",
    "constellation": "A photograph of a constellation",
    "galaxy": "A photograph of a galaxy",
    "nebula": "A photograph of a nebula",
    "planet": "A photograph of a planet",
    "star": "A photograph of a star",
}


# ============================================================================
# Pre-loaded Dataset (in memory)
# ============================================================================

def load_dataset_to_memory(root_dir: str, image_processor: CLIPImageProcessor):
    """
    Load entire dataset into memory for fast access.
    
    Returns:
        pixel_values: Tensor of shape (N, 3, 224, 224)
        labels: Tensor of shape (N,)
        class_distribution: Dict of class counts
    """
    root_dir = Path(root_dir)
    class_names = list(CLASS_DESCRIPTIONS.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    all_pixel_values = []
    all_labels = []
    class_distribution = {}
    
    print("Loading dataset into memory...")
    for class_name in tqdm(class_names, desc="Loading classes"):
        class_dir = root_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            class_distribution[class_name] = 0
            continue
        
        class_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        class_distribution[class_name] = len(class_images)
        
        for img_path in class_images:
            try:
                image = Image.open(img_path).convert('RGB')
                pixel_values = image_processor(
                    images=image,
                    return_tensors="pt"
                ).pixel_values.squeeze(0)
                
                all_pixel_values.append(pixel_values)
                all_labels.append(class_to_idx[class_name])
            except Exception as e:
                # Skip corrupted images
                continue
    
    # Stack into tensors
    pixel_values = torch.stack(all_pixel_values)
    labels = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"Loaded {len(labels)} images into memory")
    print(f"Memory usage: ~{pixel_values.element_size() * pixel_values.nelement() / 1024**3:.2f} GB")
    print(f"Class distribution: {class_distribution}")
    
    return pixel_values, labels, class_distribution


# ============================================================================
# Checkpoint Loader
# ============================================================================

class CheckpointLoader:
    """Discovers and loads CLIP model checkpoints."""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.checkpoints = self.discover_checkpoints()
    
    def discover_checkpoints(self) -> List[Tuple[int, str]]:
        """Discover all checkpoint directories and extract epoch numbers."""
        checkpoints = []
        
        for item in self.model_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-epoch-"):
                match = re.search(r'checkpoint-epoch-(\d+)', item.name)
                if match:
                    epoch = int(match.group(1))
                    checkpoints.append((epoch, str(item)))
        
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints
    
    def get_sorted_epochs(self) -> List[int]:
        """Return list of available epochs in sorted order."""
        return [epoch for epoch, _ in self.checkpoints]
    
    def get_checkpoint_path(self, epoch: int) -> Optional[str]:
        """Get checkpoint path for a specific epoch."""
        for e, path in self.checkpoints:
            if e == epoch:
                return path
        return None
    
    def load_checkpoint(self, epoch: int, device: str):
        """Load model and tokenizer from checkpoint."""
        checkpoint_path = self.get_checkpoint_path(epoch)
        if checkpoint_path is None:
            raise ValueError(f"Checkpoint for epoch {epoch} not found")
        
        model = VisionTextDualEncoderModel.from_pretrained(checkpoint_path)
        model = model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        return model, tokenizer


# ============================================================================
# CLIP Classifier
# ============================================================================

class CLIPClassifier:
    """CLIP-based image classifier."""
    
    def __init__(
        self,
        model: VisionTextDualEncoderModel,
        tokenizer: AutoTokenizer,
        device: str,
        class_descriptions: Dict[str, str],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = list(class_descriptions.keys())
        self.class_descriptions = list(class_descriptions.values())
        
        # Precompute class embeddings
        self.class_embeddings = self._compute_class_embeddings()
    
    @torch.no_grad()
    def _compute_class_embeddings(self) -> torch.Tensor:
        """Compute normalized text embeddings for all class descriptions."""
        inputs = self.tokenizer(
            self.class_descriptions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeds = self.model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        
        # L2 normalize
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds
    
    @torch.no_grad()
    def predict_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Predict class for a batch of images."""
        pixel_values = pixel_values.to(self.device)
        
        # Get image embeddings
        image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        image_embeds = F.normalize(image_embeds, dim=-1)
        
        # Compute similarity scores
        similarity = torch.matmul(image_embeds, self.class_embeddings.t())
        
        # Get predicted class
        predictions = similarity.argmax(dim=-1)
        return predictions


# ============================================================================
# Optimized Single Model Evaluator
# ============================================================================

def evaluate_single_model_optimized(
    model_name: str,
    model_dir: str,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    epochs: List[int],
    gpu_id: int,
    batch_size: int,
    results_queue: mp.Queue,
):
    """
    Evaluate a single model across ALL epochs on one GPU.
    
    Key optimization: Pre-loaded dataset in memory, iterate through epochs.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    print(f"[{model_name}] Starting evaluation on GPU {gpu_id}")
    print(f"[{model_name}] Dataset size: {len(labels)} images")
    
    # Move dataset to GPU (if it fits)
    # For 12,824 images at 224x224x3, this is ~3GB
    try:
        print(f"[{model_name}] Moving dataset to GPU...")
        pixel_values_gpu = pixel_values.to(device)
        labels_gpu = labels.to(device)
        print(f"[{model_name}] Dataset on GPU!")
    except RuntimeError as e:
        print(f"[{model_name}] Dataset too large for GPU, keeping on CPU")
        pixel_values_gpu = pixel_values
        labels_gpu = labels
    
    # Load checkpoint loader
    checkpoint_loader = CheckpointLoader(model_dir)
    available_epochs = checkpoint_loader.get_sorted_epochs()
    
    # Filter to requested epochs
    epochs_to_eval = [e for e in epochs if e in available_epochs]
    print(f"[{model_name}] Evaluating {len(epochs_to_eval)} epochs")
    
    results = []
    
    # Create dataloader
    dataset = TensorDataset(pixel_values_gpu, labels_gpu)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Data already in memory
        pin_memory=False,  # Already on GPU
    )
    
    for epoch in tqdm(epochs_to_eval, desc=f"{model_name} (GPU {gpu_id})"):
        try:
            # Load model for this epoch
            model, tokenizer = checkpoint_loader.load_checkpoint(epoch, device)
            
            # Create classifier
            classifier = CLIPClassifier(model, tokenizer, device, CLASS_DESCRIPTIONS)
            
            # Evaluate
            start_time = time.time()
            correct = 0
            total = 0
            
            with torch.cuda.amp.autocast():  # Mixed precision
                for batch_pixels, batch_labels in dataloader:
                    predictions = classifier.predict_batch(batch_pixels)
                    correct += (predictions == batch_labels).sum().item()
                    total += batch_labels.size(0)
            
            inference_time = time.time() - start_time
            accuracy = (correct / total) * 100
            
            result = EvaluationResult(
                epoch=epoch,
                model_name=model_name,
                accuracy=accuracy,
                inference_time=inference_time,
                num_samples=total,
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)
            
            print(f"[{model_name}] Epoch {epoch}: {accuracy:.2f}% ({inference_time:.1f}s)")
            
            # Clear model from GPU
            del model, classifier
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[{model_name}] Error evaluating epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Send results back
    results_queue.put((model_name, results))
    print(f"[{model_name}] Evaluation complete!")


# ============================================================================
# Parallel Evaluator
# ============================================================================

class ParallelEvaluatorOptimized:
    """Evaluates two models in parallel with pre-loaded dataset."""
    
    def __init__(
        self,
        dataset_dir: str,
        model_paths: Dict[str, str],
        batch_size: int = 1024,
    ):
        self.dataset_dir = dataset_dir
        self.model_paths = model_paths
        self.batch_size = batch_size
        
        # Pre-load dataset into memory
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.pixel_values, self.labels, self.class_distribution = load_dataset_to_memory(
            dataset_dir, image_processor
        )
    
    def evaluate_all_epochs(
        self,
        epochs: Optional[List[int]] = None,
        sample_interval: Optional[int] = None,
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate both models across specified epochs in parallel."""
        
        # Determine epochs to evaluate
        if epochs is None:
            first_model = list(self.model_paths.values())[0]
            loader = CheckpointLoader(first_model)
            epochs = loader.get_sorted_epochs()
        
        if sample_interval is not None:
            epochs = epochs[::sample_interval]
        
        print(f"\nEvaluating {len(epochs)} epochs per model")
        print(f"Batch size: {self.batch_size}")
        
        # Create multiprocessing queue
        mp.set_start_method('spawn', force=True)
        results_queue = mp.Queue()
        
        # Start parallel evaluation
        processes = []
        gpu_assignments = {"astrobert": 0, "physbert": 1}
        
        for model_name, model_path in self.model_paths.items():
            gpu_id = gpu_assignments.get(model_name, 0)
            
            p = mp.Process(
                target=evaluate_single_model_optimized,
                args=(
                    model_name,
                    model_path,
                    self.pixel_values,
                    self.labels,
                    epochs,
                    gpu_id,
                    self.batch_size,
                    results_queue,
                )
            )
            p.start()
            processes.append(p)
        
        # Collect results
        all_results = {}
        for _ in range(len(processes)):
            model_name, results = results_queue.get()
            all_results[model_name] = results
        
        # Wait for completion
        for p in processes:
            p.join()
        
        return all_results


# ============================================================================
# Results Storage & Plotting (same as before)
# ============================================================================

class ResultsStorage:
    """Stores and manages benchmark results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: Dict[str, List[EvaluationResult]], metadata: BenchmarkMetadata):
        """Save results to CSV and metadata to JSON."""
        rows = []
        for model_name, model_results in results.items():
            for r in model_results:
                rows.append(asdict(r))
        
        df = pd.DataFrame(rows)
        
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        
        json_path = self.output_dir / "benchmark_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        print(f"\nResults saved to {csv_path}")
        print(f"Metadata saved to {json_path}")
        
        return str(csv_path), str(json_path)


class PerformancePlotter:
    """Creates performance plots."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_accuracy_vs_epochs(self, results: Dict[str, List[EvaluationResult]]):
        """Create line plot showing accuracy progression."""
        plt.figure(figsize=(14, 8))
        
        colors = {'astrobert': '#2E86AB', 'physbert': '#A23B72'}
        markers = {'astrobert': 'o', 'physbert': 's'}
        
        for model_name, model_results in results.items():
            # Skip if no results for this model
            if not model_results:
                print(f"Warning: No results for {model_name}, skipping in plot")
                continue
                
            epochs = [r.epoch for r in model_results]
            accuracies = [r.accuracy for r in model_results]
            
            color = colors.get(model_name, '#333333')
            marker = markers.get(model_name, 'o')
            
            plt.plot(
                epochs, accuracies,
                label=model_name.upper(),
                color=color,
                linewidth=2,
                marker=marker,
                markersize=4,
                alpha=0.8,
            )
            
            # Mark best epoch
            best_idx = np.argmax(accuracies)
            best_epoch = epochs[best_idx]
            best_acc = accuracies[best_idx]
            
            plt.scatter(
                [best_epoch], [best_acc],
                color=color,
                s=200,
                marker='*',
                zorder=5,
                label=f'{model_name.upper()} Best: {best_acc:.2f}% (Epoch {best_epoch})'
            )
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
        plt.title('CLIP Model Performance on SpaceNet Dataset', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / "accuracy_vs_epochs.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {output_path}")
        return str(output_path)


class ReportGenerator:
    """Generates benchmark reports."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(
        self,
        results: Dict[str, List[EvaluationResult]],
        metadata: BenchmarkMetadata,
        plot_path: str,
    ):
        """Generate markdown report."""
        report_lines = [
            "# CLIP SpaceNet Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Statistics",
            "",
            f"- **Path:** {metadata.dataset_path}",
            f"- **Total Samples:** {metadata.dataset_size}",
            f"- **Batch Size:** {metadata.batch_size}",
            "",
            "### Class Distribution",
            "",
        ]
        
        for class_name, count in metadata.class_distribution.items():
            report_lines.append(f"- {class_name}: {count}")
        
        report_lines.extend(["", "## Model Performance Summary", ""])
        
        for model_name, model_results in results.items():
            # Skip if no results for this model
            if not model_results:
                report_lines.extend([
                    f"### {model_name.upper()}",
                    "",
                    "- **Status:** No results (evaluation may have failed)",
                    "",
                ])
                continue
                
            accuracies = [r.accuracy for r in model_results]
            times = [r.inference_time for r in model_results]
            best_idx = np.argmax(accuracies)
            best_result = model_results[best_idx]
            final_result = model_results[-1]
            
            report_lines.extend([
                f"### {model_name.upper()}",
                "",
                f"- **Best Accuracy:** {best_result.accuracy:.2f}% (Epoch {best_result.epoch})",
                f"- **Final Accuracy:** {final_result.accuracy:.2f}% (Epoch {final_result.epoch})",
                f"- **Mean Accuracy:** {np.mean(accuracies):.2f}%",
                f"- **Std Accuracy:** {np.std(accuracies):.2f}%",
                f"- **Avg Time per Epoch:** {np.mean(times):.1f}s",
                "",
            ])
        
        report_lines.extend([
            "## Performance Plot",
            "",
            f"![Accuracy vs Epochs]({Path(plot_path).name})",
            "",
            "## Execution Details",
            "",
            f"- **Total Evaluation Time:** {metadata.total_evaluation_time:.1f}s ({metadata.total_evaluation_time/60:.1f} min)",
            f"- **GPU Devices:** {metadata.gpu_devices}",
            "",
        ])
        
        report_content = "\n".join(report_lines)
        
        report_path = self.output_dir / "benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to {report_path}")
        return str(report_path)


# ============================================================================
# Main
# ============================================================================

def run_benchmark(args):
    """Run the optimized benchmark."""
    print("=" * 60)
    print("CLIP SpaceNet Benchmark (Optimized)")
    print("=" * 60)
    
    start_time = time.time()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_paths = {
        "astrobert": args.astrobert_dir,
        "physbert": args.physbert_dir,
    }
    
    # Create evaluator (loads dataset into memory)
    evaluator = ParallelEvaluatorOptimized(
        dataset_dir=args.dataset_dir,
        model_paths=model_paths,
        batch_size=args.batch_size,
    )
    
    # Determine epochs
    epochs = None
    if args.epochs:
        epochs = [int(e) for e in args.epochs.split(',')]
    
    # Run evaluation
    print("\nStarting parallel evaluation...")
    results = evaluator.evaluate_all_epochs(
        epochs=epochs,
        sample_interval=args.sample_interval,
    )
    
    total_time = time.time() - start_time
    
    # Create metadata
    metadata = BenchmarkMetadata(
        dataset_path=args.dataset_dir,
        dataset_size=len(evaluator.labels),
        class_distribution=evaluator.class_distribution,
        model_paths=model_paths,
        batch_size=args.batch_size,
        num_workers=0,
        gpu_devices=[0, 1],
        timestamp=datetime.now().isoformat(),
        total_evaluation_time=total_time,
    )
    
    # Save results
    storage = ResultsStorage(str(output_dir))
    storage.save_results(results, metadata)
    
    # Generate plot
    plotter = PerformancePlotter(str(output_dir))
    plot_path = plotter.plot_accuracy_vs_epochs(results)
    
    # Generate report
    reporter = ReportGenerator(str(output_dir))
    reporter.generate_markdown_report(results, metadata, plot_path)
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CLIP SpaceNet Benchmark (Optimized)")
    
    parser.add_argument(
        "--dataset_dir",
        default="/bigdisk/minhpvt/quick-test/na/ML/benchmark/data/spacenet/SpaceNet.FLARE.imam_alam",
        help="Path to SpaceNet dataset"
    )
    parser.add_argument(
        "--astrobert_dir",
        default="/bigdisk/minhpvt/quick-test/na/ML/train/outputs/physclip_astrobert",
        help="Path to AstroBERT checkpoints"
    )
    parser.add_argument(
        "--physbert_dir",
        default="/bigdisk/minhpvt/quick-test/na/ML/train/outputs/physclip_physbert",
        help="Path to PhysBERT checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        default="/bigdisk/minhpvt/quick-test/na/ML/benchmark/results_optimized",
        help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (larger for better GPU utilization)")
    parser.add_argument("--epochs", type=str, default=None, help="Specific epochs (comma-separated)")
    parser.add_argument("--sample_interval", type=int, default=None, help="Evaluate every Nth epoch")
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()

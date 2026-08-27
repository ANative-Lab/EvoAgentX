import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from datasets import load_dataset
from .benchmark import Benchmark
from .scorers import QAScorer
from ..core.logging import logger


def download_real_mm_rag_data(save_dir: str = "./data/real_mm_rag") -> str:
    """Download the REAL-MM-RAG FinReport dataset.
    
    Args:
        save_dir: Directory to save the dataset files
        
    Returns:
        str: Path to the saved dataset directory
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if dataset already exists
        dataset_path = os.path.join(save_dir, "real_mm_rag_finreport.json")
        images_dir = os.path.join(save_dir, "images")
        
        if os.path.exists(dataset_path) and os.path.exists(images_dir):
            # Quick check if images directory has content
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) > 0:
                logger.info(f"Dataset already exists at {save_dir} with {len(image_files)} images")
                return save_dir
        
        logger.info("Downloading REAL-MM-RAG FinReport dataset...")
        dataset = load_dataset("ibm-research/REAL-MM-RAG_FinReport", split="test")
        
        # Create images directory
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Process dataset: save images and create metadata
        metadata_list = []
        for i, example in enumerate(dataset):
            # Create metadata entry (without the image object)
            metadata = {
                'id': example['id'],
                'query': example['query'],
                'answer': example['answer'],
                'image_filename': example['image_filename']
            }
            
            # Add rephrase levels if they exist
            for level in ['rephrase_level_1', 'rephrase_level_2', 'rephrase_level_3']:
                if level in example and example[level]:
                    metadata[level] = example[level]
            
            metadata_list.append(metadata)
            
            # Save PIL Image if it exists
            if example['image'] is not None:
                image_filename = example['image_filename']
                image_path = os.path.join(images_dir, image_filename)
                
                # Save PIL Image
                example['image'].save(image_path)
                
                if i % 100 == 0:
                    logger.info(f"Saved {i+1}/{len(dataset)} images...")
        
        # Save metadata as JSON (without image objects)
        dataset_path = os.path.join(save_dir, "real_mm_rag_finreport.json")
        with open(dataset_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Dataset downloaded to {save_dir}")
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Images saved to: {images_dir}")
        
        return save_dir
        
    except Exception as e:
        logger.error(f"Failed to download REAL-MM-RAG dataset: {str(e)}")
        raise


class RealMMRAG(Benchmark):
    """REAL-MM-RAG FinReport benchmark for multimodal retrieval evaluation.
    
    This benchmark contains financial report pages with associated queries,
    designed to test multimodal retrieval capabilities on real-world documents.
    """
    
    def __init__(self, path: str = None, mode: str = "test", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/real_mm_rag")
        self.scorer = QAScorer(metrics=["f1", "em", "acc"])
        
        # Set up file paths before calling super().__init__ which calls _load_data
        self.dataset_file = Path(path) / "real_mm_rag_finreport.json"
        self.images_dir = Path(path) / "images"
        
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)
    
    def _load_data(self):
        """Load the dataset from JSON file."""
        if not self.dataset_file.exists():
            download_real_mm_rag_data(save_dir=self.path)
        
        try:
            with open(self.dataset_file, 'r') as f:
                self._test_data = json.load(f)
            
            logger.info(f"Loaded {len(self._test_data)} samples from REAL-MM-RAG dataset")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def _get_label(self, example: Any) -> Any:
        return example["answer"]
    
    def _get_id(self, example: Any) -> Any:
        # The upstream dataset numbers examples with an int id.
        return str(example["id"])
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return self.scorer.score(prediction=prediction, label=label)
    
    @property
    def data(self) -> List[Dict[str, Any]]:
        """Get the raw dataset."""
        return self._test_data
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Dict containing query, image_filename, answer, and rephrases
        """
        if index >= len(self._test_data):
            raise IndexError(f"Index {index} out of range for dataset size {len(self._test_data)}")
        
        sample = self._test_data[index]
        
        # Add full image path
        sample['image_path'] = str(self.images_dir / sample['image_filename'])
        
        return sample
    
    def get_samples(self, start: int = 0, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get a range of samples.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive). If None, goes to end of dataset
            
        Returns:
            List of samples
        """
        end = end or len(self._test_data)
        samples = []
        
        for i in range(start, min(end, len(self._test_data))):
            samples.append(self.get_sample(i))
            
        return samples
    
    def get_random_samples(self, n: int, seed: int = 42) -> List[Dict[str, Any]]:
        """Get n random samples from the dataset.
        
        Args:
            n: Number of samples to return
            seed: Random seed for reproducibility
            
        Returns:
            List of random samples
        """
        import random
        random.seed(seed)
        
        indices = random.sample(range(len(self._test_data)), min(n, len(self._test_data)))
        return [self.get_sample(i) for i in indices]
    
    def get_query_variations(self, sample: Dict[str, Any]) -> List[str]:
        """Get all query variations for a sample.
        
        Args:
            sample: A sample from the dataset
            
        Returns:
            List of query variations (original + 3 rephrase levels)
        """
        queries = [sample['query']]
        
        # Add rephrase levels if they exist
        for level in ['rephrase_level_1', 'rephrase_level_2', 'rephrase_level_3']:
            if level in sample and sample[level]:
                queries.append(sample[level])
                
        return queries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        total_samples = len(self._test_data)
        
        # Count samples with different rephrase levels
        has_rephrase_1 = sum(1 for s in self._test_data if s.get('rephrase_level_1'))
        has_rephrase_2 = sum(1 for s in self._test_data if s.get('rephrase_level_2'))
        has_rephrase_3 = sum(1 for s in self._test_data if s.get('rephrase_level_3'))
        
        # Get unique image files
        unique_images = set(s['image_filename'] for s in self._test_data)
        
        return {
            "total_samples": total_samples,
            "unique_images": len(unique_images),
            "samples_with_rephrase_1": has_rephrase_1,
            "samples_with_rephrase_2": has_rephrase_2,
            "samples_with_rephrase_3": has_rephrase_3,
            "avg_queries_per_image": total_samples / len(unique_images)
        }


if __name__ == "__main__":
    # Download and test the dataset
    data_dir = "./debug/data/real_mm_rag"
    
    # Download dataset
    download_real_mm_rag_data(data_dir)
    
    # Initialize benchmark
    benchmark = RealMMRAG(data_dir)
    
    # Print stats
    stats = benchmark.get_stats()
    print("REAL-MM-RAG Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample data
    print("\nSample queries:")
    samples = benchmark.get_random_samples(3)
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Image: {sample['image_filename']}")
        print(f"  Query: {sample['query']}")
        print(f"  Answer: {sample['answer']}")
        
        variations = benchmark.get_query_variations(sample)
        if len(variations) > 1:
            print(f"  Query variations: {len(variations)}")
            for j, var in enumerate(variations[1:], 1):
                print(f"    Level {j}: {var[:100]}...")

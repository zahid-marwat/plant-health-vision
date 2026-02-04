"""
Unit tests for data loading and preprocessing.

Test data pipeline functionality.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.dataset_loader import PlantVillageDatasetLoader
from src.data.preprocessor import ImagePreprocessor, ClassBalancer


class TestDatasetLoader(unittest.TestCase):
    """Test dataset loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.loader = PlantVillageDatasetLoader(self.temp_dir.name)

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_create_class_mapping(self):
        """Test class mapping creation."""
        # Create test structure
        data_dir = Path(self.temp_dir.name) / 'data'
        data_dir.mkdir()
        
        crop_dir = data_dir / 'Apple'
        crop_dir.mkdir()
        (crop_dir / 'Healthy').mkdir()
        (crop_dir / 'Scab').mkdir()
        
        class_map = self.loader.create_class_mapping(data_dir)
        
        self.assertGreater(len(class_map), 0)
        self.assertIn(0, class_map)


class TestPreprocessor(unittest.TestCase):
    """Test image preprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test image
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_image_path = Path(self.temp_dir.name) / 'test.jpg'
        
        img = Image.new('RGB', (256, 256), color='red')
        img.save(self.test_image_path)

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_load_image(self):
        """Test image loading."""
        image = ImagePreprocessor.load_image(str(self.test_image_path))
        self.assertEqual(image.shape, (256, 256, 3))

    def test_resize_image(self):
        """Test image resizing."""
        image = ImagePreprocessor.load_image(str(self.test_image_path))
        resized = ImagePreprocessor.resize_image(image, (224, 224))
        self.assertEqual(resized.shape, (224, 224, 3))

    def test_normalize_image(self):
        """Test image normalization."""
        image = ImagePreprocessor.load_image(str(self.test_image_path))
        normalized = ImagePreprocessor.normalize_image(image)
        
        # Check value range (approximately)
        self.assertTrue(np.all(normalized >= -2.5))
        self.assertTrue(np.all(normalized <= 2.5))

    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        image = ImagePreprocessor.preprocess(str(self.test_image_path))
        
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertTrue(np.all(image >= -3))
        self.assertTrue(np.all(image <= 3))


class TestClassBalancer(unittest.TestCase):
    """Test class balancer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data structure
        data_dir = Path(self.temp_dir.name) / 'data'
        data_dir.mkdir()
        
        # Create class directories with different numbers of images
        for i, count in enumerate([10, 20, 30]):
            class_dir = data_dir / f'class_{i}'
            class_dir.mkdir()
            for j in range(count):
                img = Image.new('RGB', (100, 100), color='red')
                img.save(class_dir / f'image_{j}.jpg')
        
        self.data_dir = str(data_dir)

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_get_class_weights(self):
        """Test class weight calculation."""
        weights = ClassBalancer.get_class_weights(self.data_dir)
        
        self.assertEqual(len(weights), 3)
        # More frequent class should have lower weight
        self.assertLess(weights[2], weights[0])


if __name__ == '__main__':
    unittest.main()

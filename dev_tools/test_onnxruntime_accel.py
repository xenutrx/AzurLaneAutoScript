#!/usr/bin/env python3
"""
Test onnxruntime acceleration performance on macOS.
Compare execution time with GPU/NPU acceleration enabled vs disabled.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from module.logger import logger
from module.config.config import AzurLaneConfig


def test_ocr_performance(use_gpu: bool, num_iterations: int = 10) -> dict:
    """
    Test OCR model performance with or without GPU acceleration.
    
    Args:
        use_gpu: Whether to enable GPU acceleration
        num_iterations: Number of iterations to test
        
    Returns:
        dict: Performance metrics
    """
    logger.info(f'Testing OCR with use_gpu={use_gpu}, iterations={num_iterations}')
    
    # Create dummy image for OCR
    test_image = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
    
    try:
        from rapidocr import RapidOCR, OCRVersion
        
        # Create OCR model with/without acceleration
        params = {
            "Global.use_det": False,
            "Global.use_cls": False,
            "Det.model_path": None,
            "Cls.model_path": None,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_path": str(project_root / "bin/ocr_models/zh-CN/alocr-zh-cn-v2.5.dtk.onnx"),
            "Rec.rec_keys_path": str(project_root / "bin/ocr_models/zh-CN/cn.txt"),
            "EngineConfig.onnxruntime.use_dml": use_gpu,
        }
        
        logger.info(f'Creating OCR model with params: {params}')
        model = RapidOCR(params=params)
        
        # Warm up
        logger.info('Warming up model...')
        _ = model(test_image)
        
        # Measure performance
        times = []
        logger.info(f'Running {num_iterations} iterations...')
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            result = model(test_image)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            times.append(elapsed)
            logger.info(f'  Iteration {i+1}/{num_iterations}: {elapsed*1000:.2f}ms')
        
        # Calculate statistics
        times = np.array(times)
        metrics = {
            'use_gpu': use_gpu,
            'num_iterations': num_iterations,
            'min_time': float(times.min()),
            'max_time': float(times.max()),
            'mean_time': float(times.mean()),
            'std_time': float(times.std()),
            'median_time': float(np.median(times)),
            'total_time': float(times.sum()),
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f'Error during OCR test: {e}', exc_info=True)
        raise


def main():
    """Main test function."""
    logger.info('='*60)
    logger.info('ONNX Runtime Acceleration Performance Test (macOS)')
    logger.info('='*60)
    
    # Test without acceleration
    logger.info('\n[1/2] Testing WITHOUT GPU acceleration...')
    metrics_no_gpu = test_ocr_performance(use_gpu=False, num_iterations=10)
    
    logger.info('\n' + '='*60)
    logger.info('Results WITHOUT GPU acceleration:')
    logger.info(f'  Mean time: {metrics_no_gpu["mean_time"]*1000:.2f}ms')
    logger.info(f'  Median time: {metrics_no_gpu["median_time"]*1000:.2f}ms')
    logger.info(f'  Min/Max: {metrics_no_gpu["min_time"]*1000:.2f}ms / {metrics_no_gpu["max_time"]*1000:.2f}ms')
    logger.info(f'  Std Dev: {metrics_no_gpu["std_time"]*1000:.2f}ms')
    logger.info('='*60)
    
    # Test with acceleration
    logger.info('\n[2/2] Testing WITH GPU acceleration...')
    metrics_with_gpu = test_ocr_performance(use_gpu=True, num_iterations=10)
    
    logger.info('\n' + '='*60)
    logger.info('Results WITH GPU acceleration:')
    logger.info(f'  Mean time: {metrics_with_gpu["mean_time"]*1000:.2f}ms')
    logger.info(f'  Median time: {metrics_with_gpu["median_time"]*1000:.2f}ms')
    logger.info(f'  Min/Max: {metrics_with_gpu["min_time"]*1000:.2f}ms / {metrics_with_gpu["max_time"]*1000:.2f}ms')
    logger.info(f'  Std Dev: {metrics_with_gpu["std_time"]*1000:.2f}ms')
    logger.info('='*60)
    
    # Compare results
    logger.info('\n' + '='*60)
    logger.info('COMPARISON:')
    speedup = metrics_no_gpu['mean_time'] / metrics_with_gpu['mean_time']
    improvement = (1 - metrics_with_gpu['mean_time'] / metrics_no_gpu['mean_time']) * 100
    
    logger.info(f'  Speedup: {speedup:.2f}x')
    logger.info(f'  Improvement: {improvement:.1f}%')
    
    if speedup > 1.1:
        logger.info('✓ GPU acceleration provides significant speedup!')
    elif speedup > 1.0:
        logger.info('~ GPU acceleration provides modest speedup')
    else:
        logger.info('✗ No speedup observed, might be overhead or no GPU available')
    
    logger.info('='*60)
    
    return {
        'no_gpu': metrics_no_gpu,
        'with_gpu': metrics_with_gpu,
        'speedup': speedup,
        'improvement_percent': improvement,
    }


if __name__ == '__main__':
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f'Test failed: {e}', exc_info=True)
        sys.exit(1)

#!/usr/bin/env python3
"""
Performance comparison: CoreML acceleration vs CPU-only on macOS.
Tests onnxruntime with CoreML provider vs pure CPU inference.
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


def test_coreml_performance(use_coreml: bool, num_iterations: int = 20) -> dict:
    """
    Test OCR model performance with or without CoreML acceleration.
    
    Args:
        use_coreml: Whether to enable CoreML provider
        num_iterations: Number of iterations to test
        
    Returns:
        dict: Performance metrics
    """
    try:
        import onnxruntime as ort
        
        logger.info(f'Testing with use_coreml={use_coreml}, iterations={num_iterations}')
        
        # Create dummy image
        test_image = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        
        # Model path
        model_path = project_root / "bin/ocr_models/zh-CN/alocr-zh-cn-v2.5.dtk.onnx"
        
        if not model_path.exists():
            logger.error(f'Model not found: {model_path}')
            return None
        
        # Configure providers
        if use_coreml:
            providers = [
                ('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}),
                'CPUExecutionProvider',
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        logger.info(f'Providers: {providers}')
        
        # Create session
        options = ort.SessionOptions()
        options.log_severity_level = 3  # Suppress debug logs
        
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        
        # Verify which provider is active
        active_providers = session.get_providers()
        logger.info(f'Active providers: {active_providers}')
        
        # Get model inputs
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_type = session.get_inputs()[0].type
        
        logger.info(f'Input: name={input_name}, shape={input_shape}, type={input_type}')
        
        # Prepare input with correct shape: [batch, 3, 48, width]
        # Based on model input shape requirement
        test_input = np.random.randn(1, 3, 48, 320).astype(np.float32)
        
        # Warm up
        logger.info('Warming up...')
        for _ in range(3):
            _ = session.run(None, {input_name: test_input})
        
        # Measure performance
        times = []
        logger.info(f'Running {num_iterations} iterations...')
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            output = session.run(None, {input_name: test_input})
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            times.append(elapsed)
            
            if (i + 1) % 5 == 0:
                logger.info(f'  Iteration {i+1}/{num_iterations}: {elapsed*1000:.2f}ms')
        
        # Calculate statistics
        times = np.array(times)
        metrics = {
            'provider': 'CoreML' if use_coreml else 'CPU',
            'num_iterations': num_iterations,
            'min_time': float(times.min()),
            'max_time': float(times.max()),
            'mean_time': float(times.mean()),
            'std_time': float(times.std()),
            'median_time': float(np.median(times)),
            'total_time': float(times.sum()),
            'active_providers': str(active_providers),
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f'Error during test: {e}', exc_info=True)
        return None


def main():
    """Main test function."""
    logger.info('='*70)
    logger.info('CoreML vs CPU Performance Comparison on macOS')
    logger.info('Testing onnxruntime OCR inference performance')
    logger.info('='*70)
    
    # Test CPU-only
    logger.info('\n[Test 1/2] CPU-only inference...')
    logger.info('-'*70)
    metrics_cpu = test_coreml_performance(use_coreml=False, num_iterations=20)
    
    if metrics_cpu is None:
        logger.error('CPU test failed')
        return False
    
    logger.info('\nCPU Results:')
    logger.info(f'  Provider: {metrics_cpu["provider"]}')
    logger.info(f'  Active: {metrics_cpu["active_providers"]}')
    logger.info(f'  Mean: {metrics_cpu["mean_time"]*1000:.2f}ms')
    logger.info(f'  Median: {metrics_cpu["median_time"]*1000:.2f}ms')
    logger.info(f'  Min/Max: {metrics_cpu["min_time"]*1000:.2f}ms / {metrics_cpu["max_time"]*1000:.2f}ms')
    logger.info(f'  StdDev: {metrics_cpu["std_time"]*1000:.2f}ms')
    
    # Test CoreML
    logger.info('\n[Test 2/2] CoreML + Neural Engine inference...')
    logger.info('-'*70)
    metrics_coreml = test_coreml_performance(use_coreml=True, num_iterations=20)
    
    if metrics_coreml is None:
        logger.warning('CoreML test failed, falling back to results show CPU-only performance')
        return True
    
    logger.info('\nCoreML Results:')
    logger.info(f'  Provider: {metrics_coreml["provider"]}')
    logger.info(f'  Active: {metrics_coreml["active_providers"]}')
    logger.info(f'  Mean: {metrics_coreml["mean_time"]*1000:.2f}ms')
    logger.info(f'  Median: {metrics_coreml["median_time"]*1000:.2f}ms')
    logger.info(f'  Min/Max: {metrics_coreml["min_time"]*1000:.2f}ms / {metrics_coreml["max_time"]*1000:.2f}ms')
    logger.info(f'  StdDev: {metrics_coreml["std_time"]*1000:.2f}ms')
    
    # Compare
    logger.info('\n' + '='*70)
    logger.info('PERFORMANCE COMPARISON')
    logger.info('='*70)
    
    speedup = metrics_cpu['mean_time'] / metrics_coreml['mean_time']
    improvement = (1 - metrics_coreml['mean_time'] / metrics_cpu['mean_time']) * 100
    
    logger.info(f'CoreML vs CPU Speedup: {speedup:.2f}x')
    logger.info(f'Improvement: {improvement:+.1f}%')
    
    if speedup > 1.2:
        logger.info('✓ CoreML provides significant acceleration!')
    elif speedup > 1.0:
        logger.info('~ CoreML provides modest acceleration')
    elif speedup < 0.95:
        logger.warning('⚠ CoreML slower - may need investigation or fallback to CPU')
    else:
        logger.info('= Similar performance between CoreML and CPU')
    
    logger.info('='*70)
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f'Test failed: {e}', exc_info=True)
        sys.exit(1)

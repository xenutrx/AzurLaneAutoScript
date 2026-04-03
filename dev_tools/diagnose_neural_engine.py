#!/usr/bin/env python3
"""
Diagnose whether CoreML is actually using Neural Engine (ANE) or just CPU.
Check compute units and system metrics during inference.
"""

import os
import sys
import time
import platform
import subprocess
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from module.logger import logger


def get_system_info():
    """Get system info and check for Neural Engine support."""
    logger.info('='*70)
    logger.info('System Information')
    logger.info('='*70)
    
    logger.info(f'Platform: {platform.system()}')
    logger.info(f'Platform Release: {platform.release()}')
    logger.info(f'Processor: {platform.processor()}')
    logger.info(f'Machine: {platform.machine()}')
    
    # Check for Apple Silicon
    try:
        result = subprocess.run(['sysctl', 'hw.model'], capture_output=True, text=True)
        logger.info(f'hardware.model: {result.stdout.strip()}')
    except:
        pass
    
    # Check for Neural Engine availability
    try:
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                              capture_output=True, text=True, timeout=5)
        if 'Apple' in result.stdout:
            logger.info('✓ Apple Silicon detected')
    except:
        pass


def test_coreml_compute_units():
    """Test CoreML with different compute units configurations."""
    
    logger.info('\n' + '='*70)
    logger.info('Testing CoreML Compute Units')
    logger.info('='*70)
    
    try:
        import onnxruntime as ort
        
        model_path = project_root / "bin/ocr_models/zh-CN/alocr-zh-cn-v2.5.dtk.onnx"
        
        if not model_path.exists():
            logger.error(f'Model not found: {model_path}')
            return False
        
        # Test different compute unit configurations
        configs = [
            ('ALL (CPU+GPU+ANE)', {'MLComputeUnits': 'ALL'}),
            ('CPU_AND_NEU_ENGINE (CPU+ANE)', {'MLComputeUnits': 'CPU_AND_NEU_ENGINE'}),
            ('CPU_AND_GPU (CPU+GPU)', {'MLComputeUnits': 'CPU_AND_GPU'}),
            ('CPU_ONLY', {'MLComputeUnits': 'CPU'}),
        ]
        
        test_input = np.random.randn(1, 3, 48, 320).astype(np.float32)
        
        for config_name, coreml_config in configs:
            logger.info(f'\n[Config] {config_name}')
            logger.info(f'  Config: {coreml_config}')
            
            try:
                options = ort.SessionOptions()
                options.log_severity_level = 3
                
                providers = [
                    ('CoreMLExecutionProvider', coreml_config),
                    'CPUExecutionProvider'
                ]
                
                session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
                
                active_providers = session.get_providers()
                logger.info(f'  Active Providers: {active_providers}')
                
                # Get input info
                input_name = session.get_inputs()[0].name
                
                # Measure performance (3 iterations)
                times = []
                for i in range(3):
                    start = time.perf_counter()
                    _ = session.run(None, {input_name: test_input})
                    times.append(time.perf_counter() - start)
                
                mean_time = np.mean(times)
                logger.info(f'  Mean Latency: {mean_time*1000:.2f}ms')
                
            except Exception as e:
                logger.warning(f'  Error: {e}')
    
    except ImportError as e:
        logger.error(f'Cannot import onnxruntime: {e}')
        return False
    
    return True


def check_neural_engine_usage():
    """Check if Neural Engine is being used via system metrics."""
    
    logger.info('\n' + '='*70)
    logger.info('Neural Engine Usage Check')
    logger.info('='*70)
    
    try:
        import onnxruntime as ort
        
        # Try to access CoreML runtime info
        logger.info('\n[CoreML Availability Check]')
        providers = ort.get_available_providers()
        logger.info(f'Available Execution Providers: {providers}')
        
        has_coreml = 'CoreMLExecutionProvider' in providers
        logger.info(f'CoreML Provider Available: {has_coreml}')
        
        if has_coreml:
            logger.info('✓ CoreML is available on this system')
            
            # Try to load a session and check capabilities
            model_path = project_root / "bin/ocr_models/zh-CN/alocr-zh-cn-v2.5.dtk.onnx"
            if model_path.exists():
                logger.info(f'\n[Model Analysis]')
                logger.info(f'Model: {model_path.name}')
                
                options = ort.SessionOptions()
                options.log_severity_level = 0  # Verbose
                
                providers = [
                    ('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}),
                    'CPUExecutionProvider'
                ]
                
                session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
                logger.info(f'Session created successfully')
                logger.info(f'Active Providers: {session.get_providers()}')
        else:
            logger.warning('✗ CoreML is not available on this system')
        
    except Exception as e:
        logger.error(f'Error during Neural Engine check: {e}')


def main():
    """Main diagnostic."""
    logger.info('\n' + '='*70)
    logger.info('CoreML Neural Engine Diagnostic')
    logger.info('='*70)
    
    get_system_info()
    test_coreml_compute_units()
    check_neural_engine_usage()
    
    logger.info('\n' + '='*70)
    logger.info('Diagnostic Summary')
    logger.info('='*70)
    logger.info('Notes:')
    logger.info('1. CoreML can use Neural Engine if available on Apple Silicon')
    logger.info('2. MLComputeUnits setting controls which compute units to use')
    logger.info('3. Performance may be slower due to ONNX→CoreML compilation overhead')
    logger.info('4. For optimal performance on macOS, consider native CoreML or .mlpackage models')
    logger.info('='*70)


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        logger.error(f'Diagnostic failed: {e}', exc_info=True)
        sys.exit(1)

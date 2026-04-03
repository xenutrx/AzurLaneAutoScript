#!/usr/bin/env python3
"""
Test CoreML execution provider on macOS with onnxruntime.
Demonstrates proper configuration for Neural Engine acceleration.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from module.logger import logger


def test_coreml_provider():
    """Test and demonstrate CoreML provider configuration."""
    
    logger.info('='*60)
    logger.info('Testing CoreML Execution Provider on macOS')
    logger.info('='*60)
    
    try:
        import onnxruntime as ort
        
        # Check onnxruntime version
        logger.info(f'onnxruntime version: {ort.__version__}')
        
        # List available providers
        logger.info(f'Available providers: {ort.get_available_providers()}')
        logger.info('='*60)
        
        # Create session with CoreML provider
        logger.info('\n[Test 1] Creating session WITH CoreML provider...')
        
        options = ort.SessionOptions()
        options.log_severity_level = 2  # Info level
        
        # Configure CoreML with Neural Engine support
        # 'ALL' = CPU + GPU + Neural Engine (ANE)
        # 'CPU_AND_NEU_ENGINE' = CPU + Neural Engine
        # 'CPU' = CPU only
        providers_with_coreml = [
            ('CoreMLExecutionProvider', {
                'MLComputeUnits': 'ALL',  # Enable Neural Engine (ANE)
            }),
            'CPUExecutionProvider',  # Fallback
        ]
        
        logger.info(f'Providers configuration: {providers_with_coreml}')
        
        # Try loading a model with CoreML
        try:
            # Use a simple test ONNX model
            model_path = project_root / "bin/ocr_models/zh-CN/alocr-zh-cn-v2.5.dtk.onnx"
            
            if not model_path.exists():
                logger.warning(f'Model not found: {model_path}')
                logger.info('Skipping model loading test')
            else:
                logger.info(f'Loading model: {model_path}')
                
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=providers_with_coreml
                )
                
                logger.info(f'✓ Session created successfully')
                
                # Check which providers are actually being used
                active_providers = session.get_providers()
                logger.info(f'✓ Active providers: {active_providers}')
                
                # Verify CoreML is active
                has_coreml = any('CoreML' in str(p) for p in active_providers)
                if has_coreml:
                    logger.info('✓ CoreML provider is ACTIVE (Neural Engine acceleration enabled)')
                else:
                    logger.warning('⚠ CoreML provider is NOT active (fallback to CPU)')
                
        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            logger.info('Continuing with CPU-only test...')
        
        # Test 2: CPU-only provider for comparison
        logger.info('\n[Test 2] Creating session with CPU-only provider...')
        
        providers_cpu_only = ['CPUExecutionProvider']
        logger.info(f'Providers configuration: {providers_cpu_only}')
        
        try:
            session_cpu = ort.InferenceSession(
                str(model_path) if model_path.exists() else None,
                sess_options=options,
                providers=providers_cpu_only
            ) if model_path.exists() else None
            
            if session_cpu:
                logger.info(f'✓ Session created successfully')
                active_providers_cpu = session_cpu.get_providers()
                logger.info(f'✓ Active providers: {active_providers_cpu}')
        except Exception as e:
            logger.error(f'Failed to create CPU session: {e}')
        
        # Provider details reference
        logger.info('\n[Reference] CoreML MLComputeUnits options:')
        logger.info('  "ALL"                 → CPU + GPU + Neural Engine (ANE) - Recommended for max performance')
        logger.info('  "CPU_AND_NEU_ENGINE"  → CPU + Neural Engine')
        logger.info('  "CPU_AND_GPU"         → CPU + GPU (if available)')
        logger.info('  "CPU"                 → CPU only')
        logger.info('='*60)
        
    except ImportError as e:
        logger.error(f'Failed to import onnxruntime: {e}')
        logger.info('Install with: pip install onnxruntime')
        return False
    
    return True


def main():
    """Main entry point."""
    success = test_coreml_provider()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

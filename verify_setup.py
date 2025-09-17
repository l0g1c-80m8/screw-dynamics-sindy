#!/usr/bin/env python3
"""
Environment verification script for Screw Dynamics SINDy project.
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.8)")
        return False

def check_package_import(package_name, display_name=None):
    """Check if a package can be imported."""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {display_name} ({version})")
        return True
    except ImportError as e:
        print(f"   ❌ {display_name} - {e}")
        return False

def check_core_dependencies():
    """Check core scientific computing dependencies."""
    print("\n📦 Checking core dependencies...")
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('cv2', 'OpenCV'),
    ]
    
    success_count = 0
    for package, name in packages:
        if check_package_import(package, name):
            success_count += 1
    
    return success_count == len(packages)

def check_project_modules():
    """Check project-specific modules."""
    print("\n🔧 Checking project modules...")
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    modules = [
        ('src.model', 'SINDy Model'),
        ('src.dataloader', 'Data Loader'),
        ('baseline.model', 'Baseline Models'),
    ]
    
    success_count = 0
    for module, name in modules:
        if check_package_import(module, name):
            success_count += 1
    
    return success_count == len(modules)

def check_gpu_availability():
    """Check GPU/CUDA availability."""
    print("\n🖥️  Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available - {gpu_count} GPU(s)")
            print(f"   ℹ️  Primary GPU: {gpu_name}")
            return True
        else:
            print("   ⚠️  CUDA not available (CPU only)")
            return False
    except ImportError:
        print("   ❌ PyTorch not available")
        return False

def check_directories():
    """Check required directories exist."""
    print("\n📁 Checking directory structure...")
    required_dirs = [
        'src',
        'baseline', 
        'scripts',
        'notebook',
        'data',
        '.github'
    ]
    
    success_count = 0
    current_dir = Path(__file__).parent
    
    for dirname in required_dirs:
        dir_path = current_dir / dirname
        if dir_path.exists():
            print(f"   ✅ {dirname}/")
            success_count += 1
        else:
            print(f"   ❌ {dirname}/ (missing)")
    
    return success_count == len(required_dirs)

def check_config_files():
    """Check important configuration files."""
    print("\n⚙️  Checking configuration files...")
    config_files = [
        'requirements.txt',
        'setup.py',
        'pyproject.toml',
        'LICENSE',
        'README.md'
    ]
    
    success_count = 0
    current_dir = Path(__file__).parent
    
    for filename in config_files:
        file_path = current_dir / filename
        if file_path.exists():
            print(f"   ✅ {filename}")
            success_count += 1
        else:
            print(f"   ❌ {filename} (missing)")
    
    return success_count == len(config_files)

def run_quick_test():
    """Run a quick functional test."""
    print("\n🧪 Running quick functional test...")
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Test SINDy model creation
        from src.model import SindyModel
        import torch
        
        model_config = {
            'poly_order': 2,
            'include_constant': True,
            'use_sine': False,
            'input_var_dim': 5,
            'state_var_dim': 2,
            'device': 'cpu'
        }
        
        model = SindyModel(**model_config)
        
        # Test forward pass
        x = torch.randn(3, 10, 5)  # batch_size, time_steps, input_dim
        output = model(x)
        
        if output.shape == (3, 10, 2):
            print("   ✅ SINDy model test passed")
            return True
        else:
            print(f"   ❌ SINDy model test failed - wrong output shape: {output.shape}")
            return False
            
    except Exception as e:
        print(f"   ❌ Functional test failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Screw Dynamics SINDy Environment Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("Project Modules", check_project_modules),
        ("GPU Availability", check_gpu_availability),
        ("Directory Structure", check_directories),
        ("Configuration Files", check_config_files),
        ("Functional Test", run_quick_test),
    ]
    
    passed_checks = 0
    
    for check_name, check_func in checks:
        if check_func():
            passed_checks += 1
        else:
            print(f"   ⚠️  {check_name} check had issues")
    
    print(f"\n📊 Summary: {passed_checks}/{len(checks)} checks passed")
    
    if passed_checks == len(checks):
        print("🎉 Environment is fully configured and ready!")
        return True
    elif passed_checks >= len(checks) - 2:
        print("⚠️  Environment is mostly ready with minor issues")
        return True
    else:
        print("❌ Environment has significant issues that need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Quick check script to verify ProcessPriorityManager dependencies and functionality
"""

import sys

print("="*60)
print("ProcessPriorityManager - Dependency Check")
print("="*60)

# Check Python version
print(f"\n✓ Python version: {sys.version}")

# Check dependencies
dependencies = {
    'psutil': False,
    'yaml': False
}

print("\nChecking required packages...")
try:
    import psutil
    dependencies['psutil'] = True
    print(f"✅ psutil: {psutil.__version__}")
except ImportError:
    print("❌ psutil: NOT INSTALLED")

try:
    import yaml
    dependencies['yaml'] = True
    print(f"✅ PyYAML: {yaml.__version__}")
except ImportError:
    print("❌ PyYAML: NOT INSTALLED")

# Summary
print("\n" + "="*60)
if all(dependencies.values()):
    print("✅ All dependencies installed!")
    print("\nYou can now run:")
    print("  python src/ProcessPriorityManager.py")
    print("  python tests/test_process_priority_manager.py")
else:
    print("❌ Missing dependencies detected!")
    print("\nTo install missing packages, run:")
    missing = [pkg for pkg, installed in dependencies.items() if not installed]
    if 'yaml' in missing:
        missing[missing.index('yaml')] = 'pyyaml'
    print(f"  pip install {' '.join(missing)}")

print("="*60)

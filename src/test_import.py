# test_import.py

try:
    from src.components.data_transformation import DataTransformationConfig
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")

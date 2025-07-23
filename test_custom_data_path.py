#!/usr/bin/env python3
"""
Test script to verify custom data path functionality works correctly.
This script demonstrates how to use the modified data loading with custom paths.
"""

import os
import tempfile
from data.processed import ItemData, SeqData, RecDataset

def test_custom_data_path():
    """Test that data_path parameter overrides dataset_folder."""
    
    print("Testing custom data path functionality...")
    
    # Test with ItemData
    try:
        # This should work with ML_1M dataset if data exists in default location
        default_dataset = ItemData(
            root="dataset/ml-1m", 
            dataset=RecDataset.ML_1M,
            force_process=False
        )
        print("✓ Default path loading works")
        
        # Test with custom path (will likely fail unless data exists there, but tests parameter passing)
        custom_path = "/tmp/test_data_path"
        try:
            custom_dataset = ItemData(
                root="dataset/ml-1m",  # This should be ignored
                dataset=RecDataset.ML_1M,
                data_path=custom_path,  # This should override root
                force_process=False
            )
            print("✓ Custom path parameter accepted")
        except FileNotFoundError as e:
            if custom_path in str(e):
                print(f"✓ Custom path parameter working correctly (expected FileNotFoundError: {custom_path})")
            else:
                print(f"✗ Unexpected error: {e}")
                
    except Exception as e:
        print(f"Test failed with error: {e}")
        
    # Test with SeqData
    try:
        custom_path = "/tmp/test_seq_data_path"
        try:
            seq_dataset = SeqData(
                root="dataset/ml-1m",
                dataset=RecDataset.ML_1M,
                data_path=custom_path,
                is_train=True,
                force_process=False
            )
            print("✓ SeqData custom path parameter accepted")
        except FileNotFoundError as e:
            if custom_path in str(e):
                print(f"✓ SeqData custom path parameter working correctly (expected FileNotFoundError: {custom_path})")
            else:
                print(f"✗ Unexpected SeqData error: {e}")
                
    except Exception as e:
        print(f"SeqData test failed with error: {e}")
        
    print("\nCustom data path functionality verification complete!")
    print("\nUsage examples:")
    print("1. In gin config files, set: train.data_path=\"/path/to/your/data/disk\"")
    print("2. In Python code:")
    print("   dataset = ItemData(root='fallback', data_path='/custom/path', dataset=RecDataset.AMAZON)")


if __name__ == "__main__":
    test_custom_data_path()
#!/usr/bin/env python3
"""
Debug script to identify where HDF5 sequence loading is hanging.
"""

import h5py
import numpy as np
import os
import sys
from pathlib import Path

def debug_h5_file(file_path: str):
    """Debug HDF5 file loading step by step."""
    
    print(f"🔍 Starting debug of: {file_path}")
    
    # Check file existence and basic info
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"📁 File size: {file_size / (1024**3):.2f} GB")
    
    try:
        print("🔓 Opening HDF5 file...")
        with h5py.File(file_path, 'r') as f:
            print("✅ File opened successfully")
            
            # List all keys
            print(f"🔑 Available keys: {list(f.keys())}")
            
            # Check each dataset
            for key in f.keys():
                print(f"\n📊 Analyzing dataset: {key}")
                dataset = f[key]
                print(f"   Shape: {dataset.shape}")
                print(f"   Dtype: {dataset.dtype}")
                print(f"   Size: {dataset.size}")
                
                if hasattr(dataset, 'is_variable_length'):
                    print(f"   Variable length: {dataset.is_variable_length}")
            
            # Check attributes
            if hasattr(f, 'attrs') and len(f.attrs) > 0:
                print(f"\n🏷️  Attributes: {dict(f.attrs)}")
            
            # Try loading each dataset step by step
            print("\n🚀 Testing data loading...")
            
            if 'user_ids' in f:
                print("   Loading user_ids...")
                user_ids = f['user_ids'][:]
                print(f"   ✅ Loaded user_ids: {len(user_ids)} items")
            
            if 'sequence_lengths' in f:
                print("   Loading sequence_lengths...")
                seq_lengths = f['sequence_lengths'][:]
                print(f"   ✅ Loaded sequence_lengths: {len(seq_lengths)} items")
            
            if 'sequences' in f:
                print("   Loading sequences (this might take a while)...")
                print("   Checking sequences dataset properties...")
                seq_dataset = f['sequences']
                print(f"   Sequences shape: {seq_dataset.shape}")
                print(f"   Sequences dtype: {seq_dataset.dtype}")
                
                # Try loading just the first few items
                print("   Loading first 5 sequences...")
                first_seqs = seq_dataset[:5]
                print(f"   ✅ Loaded first 5 sequences successfully")
                
                # Try loading all sequences (this is where it might hang)
                print("   ⚠️  Now attempting full sequences load...")
                sys.stdout.flush()  # Ensure output is shown
                
                sequences_data = seq_dataset[:]
                print(f"   ✅ Loaded all sequences: {len(sequences_data)} items")
                
                # Convert to list format
                print("   Converting to list format...")
                item_sequences = [seq.tolist() for seq in sequences_data]
                print(f"   ✅ Converted to lists successfully")
            
            if 'item_embeddings' in f:
                print("   Loading item_embeddings...")
                embeddings = f['item_embeddings'][:]
                print(f"   ✅ Loaded embeddings: {embeddings.shape}")
                
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Check if file path provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/home/data/sequence.h5"
    
    debug_h5_file(file_path)

if __name__ == "__main__":
    main()
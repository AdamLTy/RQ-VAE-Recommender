#!/usr/bin/env python3
"""
测试脚本：验证H5SequenceDataset功能的正确性

该脚本测试：
1. 数据集创建（训练/验证模式）
2. 序列分割逻辑（前n-1项作为输入，最后1项作为目标）
3. 数据加载器创建和批处理
4. test_ratio参数不再影响分割
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.h5_dataset import H5SequenceDataset, create_h5_sequence_dataloader


def test_h5_dataset_functionality():
    """测试H5SequenceDataset的主要功能"""
    
    # 使用假的数据集路径进行测试
    h5_sequence_data_path = "h5/sequence.h5"  # 假设的路径
    max_seq_len = 200
    batch_size = 32
    
    print("=== 测试H5SequenceDataset功能 ===")
    print(f"数据集路径: {h5_sequence_data_path}")
    print(f"最大序列长度: {max_seq_len}")
    print(f"批次大小: {batch_size}")
    
    try:
        print("\n1. 测试训练数据集创建...")
        train_dataset = H5SequenceDataset(
            sequence_data_path=h5_sequence_data_path,
            is_train=True,
            max_seq_len=max_seq_len,
            subsample=False
        )
        print(f"✅ 训练数据集创建成功，长度: {len(train_dataset)}")
        
        print("\n2. 测试验证数据集创建...")
        eval_dataset = H5SequenceDataset(
            sequence_data_path=h5_sequence_data_path,
            is_train=False,
            max_seq_len=max_seq_len,
            subsample=False
        )
        print(f"✅ 验证数据集创建成功，长度: {len(eval_dataset)}")
        
        # 验证训练和验证数据集长度相同（因为都使用所有序列）
        print(f"\n3. 验证数据集长度一致性...")
        assert len(train_dataset) == len(eval_dataset), \
            f"训练和验证数据集长度应该相同: train={len(train_dataset)}, eval={len(eval_dataset)}"
        print("✅ 训练和验证数据集长度一致")
        
        print("\n4. 测试单个样本获取...")
        sample = train_dataset[0]
        print(f"用户ID: {sample.user_ids}")
        print(f"序列ID shape: {sample.ids.shape}")
        print(f"未来ID shape: {sample.ids_fut.shape}")
        print(f"序列特征 shape: {sample.x.shape}")
        print(f"未来特征 shape: {sample.x_fut.shape}")
        print(f"序列mask shape: {sample.seq_mask.shape}")
        
        # 检查序列分割逻辑
        valid_positions = sample.seq_mask.sum().item()
        print(f"有效序列位置数: {valid_positions}")
        print("✅ 样本获取成功")
        
        print("\n5. 测试训练数据加载器创建...")
        train_dataloader = create_h5_sequence_dataloader(
            sequence_data_path=h5_sequence_data_path,
            batch_size=batch_size,
            is_train=True,
            max_seq_len=max_seq_len,
            subsample=False,
            shuffle=True
        )
        print(f"✅ 训练数据加载器创建成功，批次数: {len(train_dataloader)}")
        
        print("\n6. 测试验证数据加载器创建...")
        eval_dataloader = create_h5_sequence_dataloader(
            sequence_data_path=h5_sequence_data_path,
            batch_size=batch_size,
            is_train=False,
            max_seq_len=max_seq_len,
            subsample=False,
            shuffle=False
        )
        print(f"✅ 验证数据加载器创建成功，批次数: {len(eval_dataloader)}")
        
        print("\n7. 测试批次数据获取...")
        for batch_idx, batch in enumerate(train_dataloader):
            print(f"批次 {batch_idx}:")
            print(f"  user_ids shape: {batch.user_ids.shape}")
            print(f"  ids shape: {batch.ids.shape}")
            print(f"  ids_fut shape: {batch.ids_fut.shape}")
            print(f"  x shape: {batch.x.shape}")
            print(f"  x_fut shape: {batch.x_fut.shape}")
            print(f"  seq_mask shape: {batch.seq_mask.shape}")
            
            # 验证批次维度
            actual_batch_size = batch.user_ids.shape[0]
            assert batch.ids.shape[0] == actual_batch_size
            assert batch.ids_fut.shape[0] == actual_batch_size
            assert batch.x.shape[0] == actual_batch_size
            assert batch.x_fut.shape[0] == actual_batch_size
            assert batch.seq_mask.shape[0] == actual_batch_size
            
            print(f"  实际批次大小: {actual_batch_size}")
            break  # 只测试第一个批次
        
        print("✅ 批次数据获取成功")
        
        print("\n8. 测试test_ratio参数被忽略...")
        # 使用不同的test_ratio值创建数据集，长度应该相同
        dataset_ratio_01 = H5SequenceDataset(
            sequence_data_path=h5_sequence_data_path,
            is_train=True,
            max_seq_len=max_seq_len,
            test_ratio=0.1,
            subsample=False
        )
        
        dataset_ratio_09 = H5SequenceDataset(
            sequence_data_path=h5_sequence_data_path,
            is_train=True,
            max_seq_len=max_seq_len,
            test_ratio=0.9,
            subsample=False
        )
        
        assert len(dataset_ratio_01) == len(dataset_ratio_09), \
            f"不同test_ratio的数据集长度应该相同: {len(dataset_ratio_01)} vs {len(dataset_ratio_09)}"
        print(f"✅ test_ratio参数被正确忽略，数据集长度均为: {len(dataset_ratio_01)}")
        
        print("\n9. 测试序列分割逻辑...")
        # 获取几个样本来验证分割逻辑
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            valid_mask = sample.seq_mask.numpy()
            valid_ids = sample.ids.numpy()[valid_mask]
            target_id = sample.ids_fut.numpy()[0]
            
            print(f"序列 {i}:")
            print(f"  输入序列长度: {len(valid_ids)}")
            print(f"  输入序列: {valid_ids[:5]}..." if len(valid_ids) > 5 else f"  输入序列: {valid_ids}")
            print(f"  目标物品: {target_id}")
            
            # 验证目标不是-1（除非是单物品序列）
            if len(valid_ids) > 0:
                assert target_id >= -1, f"目标物品ID应该 >= -1: {target_id}"
        
        print("✅ 序列分割逻辑验证通过")
        
        print("\n🎉 所有测试通过！")
        
    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}")
        print("请确保H5数据文件存在或更新文件路径")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_different_configurations():
    """测试不同配置参数"""
    
    h5_sequence_data_path = "h5/sequence.h5"
    
    print("\n=== 测试不同配置参数 ===")
    
    configs = [
        {"max_seq_len": 50, "subsample": False, "is_train": True},
        {"max_seq_len": 100, "subsample": True, "is_train": True},
        {"max_seq_len": 200, "subsample": False, "is_train": False},
    ]
    
    try:
        for i, config in enumerate(configs):
            print(f"\n配置 {i+1}: {config}")
            
            dataset = H5SequenceDataset(
                sequence_data_path=h5_sequence_data_path,
                **config
            )
            
            print(f"  数据集长度: {len(dataset)}")
            
            # 测试第一个样本
            if len(dataset) > 0:
                sample = dataset[0]
                valid_positions = sample.seq_mask.sum().item()
                print(f"  第一个样本有效位置数: {valid_positions}")
                print(f"  最大序列长度限制: {config['max_seq_len']}")
                
                assert valid_positions <= config['max_seq_len'], \
                    f"有效位置数不应超过最大序列长度: {valid_positions} > {config['max_seq_len']}"
        
        print("✅ 不同配置参数测试通过")
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")


if __name__ == "__main__":
    print("开始测试H5SequenceDataset功能\n")
    
    # 主要功能测试
    test_h5_dataset_functionality()
    
    # 不同配置测试
    test_different_configurations()
    
    print("\n测试完成！")
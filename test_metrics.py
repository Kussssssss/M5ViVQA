#!/usr/bin/env python3
"""
Test script để kiểm tra metrics function hoạt động đúng.
"""

import sys
import os

# Thêm đường dẫn hiện tại vào sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_metrics_import():
    """Test việc import metrics module."""
    try:
        from openvivqa.evaluation.metrics import compute_vqa_metrics
        print("✓ Import metrics thành công")
        return True
    except Exception as e:
        print(f"✗ Import metrics thất bại: {e}")
        return False

def test_metrics_computation():
    """Test việc tính toán metrics."""
    try:
        from openvivqa.evaluation.metrics import compute_vqa_metrics
        
        # Test data đơn giản
        predictions = [
            "Đây là câu trả lời đầu tiên",
            "Câu trả lời thứ hai",
            "Câu trả lời thứ ba"
        ]
        
        references = [
            "Đây là câu trả lời đầu tiên",
            "Câu trả lời thứ hai", 
            "Câu trả lời thứ ba"
        ]
        
        # Tính metrics
        metrics = compute_vqa_metrics(predictions, references)
        
        print("✓ Tính metrics thành công")
        print(f"  Metrics: {metrics}")
        
        # Kiểm tra các keys cần thiết
        required_keys = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL", "cider"]
        for key in required_keys:
            if key not in metrics:
                print(f"✗ Thiếu key: {key}")
                return False
            if not isinstance(metrics[key], (int, float)):
                print(f"✗ Key {key} không phải số: {type(metrics[key])}")
                return False
        
        print("✓ Tất cả metrics keys đều hợp lệ")
        return True
        
    except Exception as e:
        print(f"✗ Tính metrics thất bại: {e}")
        return False

def test_empty_input():
    """Test với input rỗng."""
    try:
        from openvivqa.evaluation.metrics import compute_vqa_metrics
        
        metrics = compute_vqa_metrics([], [])
        
        print("✓ Xử lý input rỗng thành công")
        print(f"  Metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Xử lý input rỗng thất bại: {e}")
        return False

def main():
    """Chạy tất cả tests."""
    print("Bắt đầu test metrics...")
    print("-" * 50)
    
    tests = [
        test_metrics_import,
        test_metrics_computation,
        test_empty_input
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} gặp lỗi: {e}")
        print()
    
    print("-" * 50)
    print(f"Kết quả: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Tất cả tests đều thành công!")
        return 0
    else:
        print("❌ Một số tests thất bại")
        return 1

if __name__ == "__main__":
    sys.exit(main())

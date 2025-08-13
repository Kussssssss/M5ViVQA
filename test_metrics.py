"""
Test script để kiểm tra hàm compute_vqa_metrics hoạt động đúng.
"""

import sys
import os

# Thêm thư mục gốc vào Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openvivqa.evaluation.metrics import compute_vqa_metrics

def test_metrics():
    """Test hàm compute_vqa_metrics với dữ liệu mẫu."""
    
    # Dữ liệu test đơn giản
    predictions = [
        "Màu đỏ",
        "Xanh lá cây", 
        "Xanh dương"
    ]
    
    references = [
        "Màu đỏ",
        "Xanh lá",
        "Xanh dương"
    ]
    
    print("Testing compute_vqa_metrics...")
    print(f"Predictions: {predictions}")
    print(f"References: {references}")
    
    try:
        metrics = compute_vqa_metrics(predictions, references)
        print("\nMetrics computed successfully:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Kiểm tra các giá trị hợp lý
        assert 0 <= metrics["bleu1"] <= 1, f"BLEU-1 should be between 0 and 1, got {metrics['bleu1']}"
        assert 0 <= metrics["meteor"] <= 1, f"METEOR should be between 0 and 1, got {metrics['meteor']}"
        assert 0 <= metrics["rougeL"] <= 1, f"ROUGE-L should be between 0 and 1, got {metrics['rougeL']}"
        assert metrics["cider"] >= 0, f"CIDEr should be non-negative, got {metrics['cider']}"
        
        print("\n✅ All metrics are within expected ranges!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_metrics()
    sys.exit(0 if success else 1)

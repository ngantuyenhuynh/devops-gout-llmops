#!/bin/bash
MODELS=("qwen2:1.5b")

for MODEL in "${MODELS[@]}"; do
    echo "================================================="
    echo "🚀 ĐANG KÍCH HOẠT MÔ HÌNH: $MODEL"
    echo "================================================="
    
    # [ĐÃ SỬA LỖI]: Tìm dòng có chữ EVAL_MODEL_NAME, nhảy xuống dòng tiếp theo và thay thế giá trị
    sed -i "/name: EVAL_MODEL_NAME/{n;s/value: .*/value: $MODEL/}" k8s/evaluation-job/job.yaml
    
    kubectl delete -f k8s/evaluation-job/job.yaml --ignore-not-found > /dev/null 2>&1
    kubectl apply -f k8s/evaluation-job/job.yaml
    
    echo "⏳ Đang chờ $MODEL giải 58 câu (có thể mất vài phút)..."
    kubectl wait --for=condition=complete job/evaluation-job -n gout-eval --timeout=30m
    
    kubectl logs job/evaluation-job -n gout-eval > "result_${MODEL}.json"
    echo "✅ Đã lưu thành công: result_${MODEL}.json"
    echo ""
done

echo "🎉 HOÀN THÀNH XUẤT SẮC! CẢ 3 MODEL ĐÃ CHẠY XONG!"

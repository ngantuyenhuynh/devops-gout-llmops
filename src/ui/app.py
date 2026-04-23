import streamlit as st
import requests

# 1. Cấu hình trang Web
st.set_page_config(page_title="Gout AI Consultant", page_icon="⚕️", layout="centered")

st.title("⚕️ Trợ lý Y khoa Bệnh Gút (Gout-LLM)")
st.markdown("Hệ thống tư vấn tích hợp RAG và LLMOps cho Sinh viên Y & Bệnh nhân.")

# 2. Sidebar để cấu hình (Chọn mô hình)
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển (DevOps)")
    selected_model = st.selectbox(
        "🧠 Chọn Bác sĩ AI (Model):",
        ["qwen2:1.5b", "vistral:7b", "vinallama:7b", "phogpt:4b"]
    )
    st.info("💡 Điểm Evaluation (RAGAS) của các mô hình này được tự động chấm và cập nhật trên Grafana Dashboard thông qua luồng CI/CD.")
    st.success("Trạng thái RAG: Đang hoạt động")

# 3. Khởi tạo bộ nhớ lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Xử lý khi người dùng nhập câu hỏi mới
if prompt := st.chat_input("VD: Người bệnh gút có được uống sữa đậu nành không?"):
    
    # Hiển thị tin nhắn của User
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Gọi API Orchestrator (K8s Backend)
    with st.chat_message("assistant"):
        with st.spinner(f"Bác sĩ {selected_model} đang tra cứu phác đồ QĐ 361/QĐ-BYT..."):
            try:
                # Địa chỉ Backend đang chạy (cổng 8000)
                API_URL = "http://eval-orchestrator-service:8000/ask"
                payload = {"question": prompt, "model_name": selected_model}
                
                response = requests.post(API_URL, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()

                if data.get("error"):
                    raise RuntimeError(str(data["error"]))

                answer = data.get("answer", "Xin lỗi, tôi không thể sinh văn bản lúc này.")
                sources = data.get("sources", [])
                
                # In câu trả lời
                st.markdown(answer)
                
                # In nguồn trích dẫn
                if sources:
                    st.caption(f"📚 **Nguồn tài liệu truy xuất (RAG):** {', '.join(sources)}")
                    
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"🔴 Lỗi kết nối đến Backend K8s: {str(e)}")

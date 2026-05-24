import streamlit as st
import requests

# 1. Cấu hình trang Web
st.set_page_config(page_title="Gout-LLMOps AI Consultant", page_icon="⚕️", layout="centered")

st.title("⚕️ Gout-LLMOps AI Consultant")
st.markdown("Hệ thống hỏi - đáp y khoa ứng dụng RAG, được kiểm định an toàn qua quy trình CI/CD/CE tự động.")

# 2. Sidebar để cấu hình (Chọn mô hình)
with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    selected_model = st.selectbox(
        "🧠 Chọn Model:",
        ["qwen2.5:1.5b", "qwen2:1.5b"]
    )
    rag_enabled = st.checkbox("Kích hoạt mode RAG", value=True)
    if rag_enabled:
        st.success("Trạng thái RAG: Đang hoạt động")
    else:
        st.warning("Trạng thái RAG: Đã tắt")

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

    # 5. Gọi API Orchestrator (K8s Backend)
    with st.chat_message("assistant"):
        with st.spinner(f"Đang sinh câu trả lời từ mô hình {selected_model}..."):
            try:
                import json
                # Địa chỉ Backend đang chạy (cổng 8000)
                API_URL = "http://eval-orchestrator-service:8000/ask/stream"
                
                # Thiết lập payload với RAG hoặc không
                payload = {"question": prompt, "model_name": selected_model}
                if rag_enabled:
                    payload["rag"] = True  # Kích hoạt RAG trong payload

                response = requests.post(API_URL, json=payload, timeout=120, stream=True)
                response.raise_for_status()

                # Kiểm tra nếu response trả về JSON lỗi nguyên khối
                if response.headers.get("content-type") == "application/json":
                    data = response.json()
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))

                source_box = []
                def generate_chunks():
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if data.get("type") == "error":
                                st.error(data["error"])
                                break
                            elif data.get("type") == "sources":
                                source_box.append(data.get("sources", []))
                            elif data.get("type") == "chunk":
                                yield data["content"]

                # st.write_stream sẽ tự động in từng chunk ra màn hình
                answer = st.write_stream(generate_chunks())

                # Lấy sources ra sau khi stream kết thúc
                sources = source_box[0] if source_box else []
                # In nguồn trích dẫn (chỉ khi RAG được kích hoạt)
                if rag_enabled and sources:
                    st.caption(f"**Nguồn tài liệu truy xuất (RAG):** {', '.join(sources)}")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"🔴 Lỗi kết nối đến Backend K8s: {str(e)}")

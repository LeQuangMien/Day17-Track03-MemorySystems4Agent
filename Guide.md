# Hướng dẫn từng bước để hoàn thành Giai đoạn 2, Track 3, Day 17

Tài liệu này hướng dẫn các bạn hoàn thành bài lab của Day 17 theo đúng thứ tự hợp lý. Mục tiêu là làm được bài trong `src/` với một lộ trình rõ ràng từ setup, memory layer, đến benchmark.

## Bước 1. Đọc cấu trúc repo

Trước khi code, các bạn cần hiểu repo đang chia trách nhiệm như thế nào.

Trước khi đọc code, hãy dựng môi trường trước:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

- `src/` là phần các bạn phải hoàn thiện
- `data/` là input benchmark dùng chung

Việc đầu tiên nên làm là đọc nhanh các file sau:

1. `src/README.md`
2. `src/config.py`
3. `src/memory_store.py`
4. `src/agent_baseline.py`
5. `src/agent_advanced.py`
6. `src/benchmark.py`
7. `src/test_agents.py`

## Bước 2. Hoàn thiện cấu hình chung

File cần làm: `src/config.py`

Các bạn cần triển khai:

- cấu hình đường dẫn repo, `data/`, `state/`
- ngưỡng compact memory
- số message cần giữ lại sau khi compact
- cấu hình provider cho model chính và model judge

Provider cần hỗ trợ trong thiết kế:

- `openai`
- `custom`
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Kết quả mong đợi:

- có một `LabConfig`
- có `load_config()` trả về config hoàn chỉnh

## Bước 3. Làm lớp memory cơ bản

File cần làm: `src/memory_store.py`

Đây là phần cốt lõi của cả track. Các bạn cần hoàn thiện ba ý:

- ước lượng token đơn giản
- quản lý `User.md`
- compact memory cho hội thoại dài

### 3.1. `estimate_tokens()`

Không cần quá chính xác theo tokenizer thực. Một estimator heuristic ổn định là đủ để benchmark offline.

### 3.2. `UserProfileStore`

Agent advanced cần có nơi lưu thông tin bền vững của người dùng. Tối thiểu phải có:

- `path_for()`
- `read_text()`
- `write_text()`
- `edit_text()`
- `file_size()`

Nếu muốn tốt hơn, có thể thêm:

- `facts()`
- `upsert_fact()`

### 3.3. `extract_profile_updates()`

Các bạn cần trích được một số facts ổn định từ message người dùng, ví dụ:

- tên
- nơi ở
- nghề nghiệp
- phong cách trả lời mong muốn
- sở thích hoặc mối quan tâm kỹ thuật

### 3.4. `CompactMemoryManager`

Yêu cầu:

- giữ một số message gần nhất
- khi token vượt ngưỡng thì tóm tắt message cũ
- lưu số lần compaction để benchmark

## Bước 4. Hoàn thiện Baseline Agent

File cần làm: `src/agent_baseline.py`

Baseline agent phải thật sự “ngây thơ” ở mức hợp lý:

- nhớ trong cùng thread
- không có `User.md`
- không nhớ facts dài hạn qua thread mới

Các bạn cần hoàn thiện:

- `reply()`
- `_reply_offline()`
- `token_usage()`
- `prompt_token_usage()`
- `_maybe_build_langchain_agent()`

Điều quan trọng:

- baseline không nên giả vờ có long-term memory
- baseline phải là mốc so sánh công bằng cho advanced agent

## Bước 5. Hoàn thiện Advanced Agent

File cần làm: `src/agent_advanced.py`

Advanced agent phải có đủ ba lớp memory:

1. short-term memory
2. persistent memory bằng `User.md`
3. compact memory

Các bạn cần hoàn thiện:

- `reply()`
- `_reply_offline()`
- `_estimate_prompt_context_tokens()`
- `_offline_response()`
- `_maybe_build_langchain_agent()`

Khi làm phần này, hãy luôn tự hỏi:

- fact nào nên lưu vào `User.md`?
- fact nào chỉ nên giữ tạm trong thread?
- summary có đang làm mất thông tin quan trọng không?

## Bước 6. Làm benchmark

File cần làm: `src/benchmark.py`

Benchmark của các bạn phải có **hai phần**:

### 6.1. Standard Benchmark

Input:

- `data/conversations.json`

Mục tiêu:

- đo khả năng recall qua nhiều hội thoại bình thường
- so sánh baseline và advanced

### 6.2. Long-Context Stress Benchmark

Input:

- `data/advanced_long_context.json`

Mục tiêu:

- làm lộ tác động của compact memory khi hội thoại rất dài
- chứng minh rằng compact tối ưu chủ yếu ở `prompt tokens processed`

### 6.3. Các cột benchmark bắt buộc

- `Agent tokens only`
- `Prompt tokens processed`
- `Cross-session recall`
- `Response quality`
- `Memory growth (bytes)`
- `Compactions`

## Bước 7. Làm test pass

File cần làm: `src/test_agents.py`

Tối thiểu các bạn cần có test cho:

- `User.md` read/write/edit
- compact memory trigger
- cross-session recall
- prompt load giảm xuống khi compact hoạt động

Đây là phần giúp phân biệt giữa “trông có vẻ chạy được” và “đã kiểm chứng được hành vi memory”.

## Bước 8. Viết phần phân tích kết quả

Sau khi benchmark chạy được, các bạn cần viết ngắn gọn:

- vì sao advanced có recall tốt hơn baseline
- vì sao advanced có thể tốn hơn ở hội thoại ngắn
- vì sao compact giúp advanced có lợi thế ở hội thoại dài
- file memory tăng trưởng ra sao và rủi ro gì đi kèm

## Bước 9. Bonus nếu muốn lên mức 90-100

Các hướng bonus có giá trị:

- confidence threshold trước khi ghi vào `User.md`
- memory decay cho thông tin cũ
- entity extraction có cấu trúc hơn
- chiến lược tránh lưu sai khi người dùng đặt câu hỏi thay vì cung cấp fact

## Thứ tự triển khai ngắn gọn

Nếu muốn đi nhanh mà không lạc hướng, hãy theo đúng thứ tự này:

1. `config.py`
2. `memory_store.py`
3. `agent_baseline.py`
4. `agent_advanced.py`
5. `benchmark.py`
6. `test_agents.py`
7. phân tích kết quả

Nếu các bạn giữ được thứ tự này, bài sẽ ổn định hơn rất nhiều so với việc nhảy thẳng vào benchmark khi memory layer còn chưa chắc chắn.

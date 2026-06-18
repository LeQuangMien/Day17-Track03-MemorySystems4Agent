# Rubric chấm Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Rubric này dùng để đánh giá bài làm của Day 17, bao gồm cả phần triển khai kỹ thuật lẫn phần hiểu đúng trade-off của memory system.

## 0-60 điểm: Hoàn thành triển khai cơ bản

Để đạt mức này, bài làm cần có đủ các phần sau:

- có `Baseline Agent` chỉ nhớ trong cùng thread
- có `Advanced Agent` với `User.md` bền vững
- có compact memory hoặc cơ chế nén lịch sử tương đương
- có benchmark dataset tiếng Việt
- có README, Guide, và cấu trúc repo rõ ràng

Các lỗi trừ điểm mạnh trong vùng này:

- baseline vẫn nhớ được qua session mới theo cách không mong muốn
- advanced không lưu được `User.md`
- compact memory không thực sự kích hoạt
- benchmark chỉ in kết quả nhưng không phản ánh đúng sự khác biệt giữa hai agent

## 60-75 điểm: Có benchmark chạy được và có test cốt lõi

Để vượt mốc cơ bản, bài cần thêm:

- benchmark chạy cùng input cho cả baseline và advanced
- có ít nhất một bài test cho `User.md`
- có ít nhất một bài test cho compact trigger
- có ít nhất một bài test cho cross-session recall

Kết quả benchmark tối thiểu cần có các cột:

- `Agent tokens only`
- `Prompt tokens processed`
- `Cross-session recall`
- `Response quality`
- `Memory growth (bytes)`
- `Compactions`

## 75-90 điểm: Phân tích được tác động thật của compact memory

Để vào nhóm này, bài cần không chỉ chạy được mà còn cho thấy hiểu đúng bản chất hệ thống.

Kỳ vọng:

- có `Standard Benchmark`
- có `Long-Context Stress Benchmark`
- stress benchmark đủ dài để làm lộ chi phí ngữ cảnh của baseline
- có phân tích vì sao compact không phải lúc nào cũng thắng ở hội thoại ngắn
- có giải thích vì sao compact chủ yếu tối ưu `prompt tokens processed`

Điểm cao hơn trong dải này nếu:

- dữ liệu benchmark tự nhiên, có correction, follow-up, open thread
- có tách bạch rõ `short-term`, `persistent`, `compact`
- có nhận xét về rủi ro memory file phình to hoặc lưu sai fact

## 90-100 điểm: Có phần bonus thật sự hữu ích

Để vào nhóm cao nhất, bài nên có ít nhất một mở rộng có giá trị kỹ thuật thực tế:

- `Confidence threshold`: chỉ ghi vào `User.md` khi đủ chắc chắn
- `Memory decay`: thông tin cũ giảm ưu tiên theo thời gian hoặc theo tần suất nhắc lại
- `Entity extraction`: facts được tách thành field có cấu trúc tốt hơn
- `Conflict handling`: khi có correction mới, agent cập nhật đúng và không giữ đồng thời fact cũ sai

Điểm 100 nên dành cho bài vừa có bonus, vừa giải thích rõ:

- bonus đó giải quyết vấn đề gì
- nó cải thiện recall hoặc token cost như thế nào
- nó tạo thêm rủi ro gì cho hệ thống

## Tiêu chí đánh giá chất lượng bài làm

Ngoài việc “đủ tính năng”, bài tốt cần có các dấu hiệu sau:

- code tách lớp rõ ràng
- benchmark đọc được và có ý nghĩa
- naming nhất quán giữa baseline và advanced
- dữ liệu benchmark phù hợp với mục tiêu memory
- test phản ánh đúng hành vi hệ thống thay vì chỉ test happy path

## Một bài rất tốt thường sẽ có câu chuyện rõ ràng

Reviewer nên nhìn thấy được luồng logic này:

1. baseline không nhớ dài hạn
2. advanced thêm `User.md` nên recall tăng
3. hội thoại dài làm prompt cost tăng mạnh
4. compact memory kéo chi phí ngữ cảnh xuống
5. hệ thống mạnh hơn nhưng cũng phức tạp hơn và cần guardrail tốt hơn

Nếu bài thể hiện rõ được logic trên bằng code, test, benchmark, và phần phân tích, đó là một bài rất mạnh cho Track 3.

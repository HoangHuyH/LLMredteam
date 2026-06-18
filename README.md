# CPA — Thử nghiệm đầu độc CTI cho AI pentest (sandbox)

> **Đọc [`SAFETY.md`](SAFETY.md) trước.** Toàn bộ chạy trong sandbox đóng kín. Không gửi gì ra ngoài.

## Là gì

Bộ thử nghiệm **phòng thủ**. Thả tin tình báo an ninh (CTI) **giả** vào một kho nội bộ kín, cho một AI pentest đọc qua RAG, rồi đo xem nó có bị dẫn sai không (lập kế hoạch lệch, báo lỗ hổng không có thật). Sau đó thử bộ lọc để chặn.

Chạy hoàn toàn trong sandbox. Không gửi gì ra ngoài. Không có code tấn công thật.

---

## 3 câu hỏi

- **RQ1**: CTI giả đúng ngữ cảnh mục tiêu (Group B) có hại hơn CTI giả vu vơ (Group A) không?
- **RQ2**: Thêm RL (PPO) có giúp tấn công vừa mạnh vừa khó bị phát hiện hơn baseline DREAM không?
- **RQ3**: Yếu tố nào ảnh hưởng mạnh nhất? (hồi quy 4 chiều: target_relevance, stealth, CFR, PDS)

---

## Một vòng chạy

```
policy chọn cách thả CTI giả
  → ghi vào kho mock (Chroma + MiniLM)
  → victim đọc (RAG) + lập kế hoạch
  → đọc output victim → tính điểm (PDS / FPR / stealth)
  → cập nhật policy (PPO)
```

Env có **reactive detection-heat**: publish nhiều → heat tăng → risk tăng → reward giảm.

---

## Hai loại victim

| Victim | Dùng khi | Chú ý |
|--------|----------|-------|
| `rule_based` | Train PPO, RQ1 smoke, defense | Nhanh, tất định. **CẢNH BÁO**: victim này được lập trình để hễ tin CTI giả là sập — số từ nó là *trần trên*, KHÔNG phải bằng chứng tấn công thật. |
| `hf` (Qwen2.5-7B) | Validate RQ1 trên victim LLM thật | Chậm (~5 phút/episode). Hiện trạng: **tấn công chưa tách khỏi nhiễu sampling LLM** (nopoison arm đạt PDS ~0.33 do nhiễu). Dùng `ASR_calibrated_%` (mean+2σ noise floor) thay vì raw ASR. |

---

## Số liệu nói gì (trung thực)

- Trên **rule_based**: gần 100% "thành công" — nhưng do victim bị lập trình trước.
- Trên **Qwen thật**: xem `RESULTS.md`. Dùng `ASR_calibrated_%` và `pds_net_B` (đã trừ nhiễu nền).
- **Defense chặn template poison ~100%** — nhưng template quá dễ nhận dạng, là trần trên.
- Kết quả t-test kèm **Bonferroni correction** (n_tests=3 cho RQ1, n_tests=5 cho RQ2).

---

## Thật / giả trong repo

| Thứ | Trạng thái |
|-----|-----------|
| CTI thật | CASIE + CyEnts (thật, clone tự động) |
| GFCTI-Finance | Loader sẵn sàng. **Chưa có data gốc** → đang dùng 5 dòng synthetic (cảnh báo to khi chạy). |
| Victim PentestGPT V2 | Chưa có → dùng Qwen2.5-7B làm proxy |
| Số liệu kết quả | Xem `RESULTS.md` (cập nhật sau mỗi Kaggle run) |

---

## Chạy

### Local smoke (không cần GPU)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m data.build_corpus
python -m experiments.run_rq --rq all --episodes 6 --max-targets 4
```

### Kaggle (chính)
Dán `kaggle/todo.py` vào 1 cell → Run. Code sửa phải **commit + push lên `kaggle-run`** — driver `git reset --hard origin/kaggle-run`.

Cờ quan trọng ở đầu file:
- `SESSION=1` (rule_based, ~4h) → lưu output → `SESSION=2` (Qwen thật, RESUME từ S1)
- `SCALE`: `qwen1s`(4tgt/5ep) | `smoke`(3tgt/8ep) | `medium`(8tgt/15ep)
- `EVAL_VICTIM`: `rule_based` | `hf`(Qwen, chỉ dùng với `qwen1s`/`smoke`)

---

## An toàn

Xem [`SAFETY.md`](SAFETY.md). Không publish ra nền tảng thật. Không có module khai thác.

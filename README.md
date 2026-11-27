# ชุดสคริปต์ประเมินผลเพิ่มเติมตามความเห็นผู้ทรงคุณวุฒิ

ใช้กับระบบ MediaPipe → GRU (หน้าต่าง 15 เฟรม) → กฎคงอยู่ 3 เฟรม + เกณฑ์ความเร็วสะโพก

| ไฟล์ | ทำอะไร | ตอบความเห็น |
|---|---|---|
| `loso.py` | ประเมิน GRU แบบ LOSO, แบบแยกวิดีโอ และแบบสุ่ม 80/20 (แบบเดิม) แล้วเทียบกัน | R1 ข้อ 1, R2 ข้อ 1, R3 |
| `predict_trials.py` | สร้างค่าความน่าจะเป็นของ GRU สำหรับวิดีโอทดสอบที่บันทึกแยก (เช่น 150 สถานการณ์, ADL 80 ครั้ง) โดยไม่ใช้โมเดลที่เคยเห็นผู้ทดลองคนนั้น | ใช้ประกอบ lead time / ablation |
| `lead_time.py` | วัดเวลาเตือนล่วงหน้าก่อนกระแทก (ms) + ตัวช่วยกำหนดเฟรมกระแทก | R1 ข้อ 2, R2 ข้อ 2, R3 |
| `ablation.py` | ตาราง ablation ของกฎเสริม + McNemar test + การวิเคราะห์ความไวของ threshold + เลือก threshold แบบ LOSO | R2 ข้อ 5, R3 (เหตุผลการเลือก threshold) |
| `benchmark_edge.py` | วัด latency แยกขั้นตอน, FPS, RAM, CPU, อุณหภูมิ บน Raspberry Pi 5 / Jetson | R1 ข้อ 2, R3 |
| `config.py` | ค่าตั้งทั้งหมด (**ต้องแก้ให้ตรงกับงานก่อนใช้**) | – |
| `data.py`, `decision.py`, `model.py`, `utils.py` | โมดูลที่สคริปต์ข้างบนเรียกใช้ | – |
| `make_demo_data.py` | สร้างข้อมูลจำลองไว้ทดลองรันเท่านั้น (ตัวเลขที่ได้ไม่มีความหมาย) | – |
| `prepare_open_data.py` | แปลงชุดข้อมูลเปิด (OmniFall: UP-Fall, CMDFall, CAUCAFall, Le2i, GMDCSA24 …, URFD) เป็น frames.csv / trials.csv ดู **README_open_data.md** | R1 ข้อ 1, R2 ข้อ 1, 4 |
| `get_public_data.py` | ดาวน์โหลด URFD และ label ของ OmniFall อัตโนมัติ แล้วพิมพ์คำสั่งแปลงข้อมูลขั้นต่อไป | R1 ข้อ 1, R2 ข้อ 1 |
| `datasets/gmdcsa24/` | สร้างชุด GMDCSA24 พร้อมอัปโหลดเข้าเว็บแอป (frames, trials, predictions แบบ LOSO, ภาพตรวจจุดกระแทก) ดู `datasets/gmdcsa24/README_TH.md` | R1 ข้อ 1–2, R2 ข้อ 1–2 |

## เว็บแอป (https://fallriskpredict.vercel.app)

อัปโหลด `frames.csv`, `trials.csv` และไฟล์ค่าทำนายของ GRU (`oof_predictions.csv` จาก `loso.py` หรือไฟล์ที่มีคอลัมน์ `p_pos`) แล้วตั้งค่ากฎตัดสินใจ เว็บจะคำนวณให้ทันที
- KPI, histogram ของ lead time, ตารางแยกตามสถานการณ์
- ตาราง ablation พร้อม McNemar test
- heatmap ความไวของ threshold, กราฟ trade-off และการเลือก threshold แบบ LOSO
- กราฟไทม์ไลน์ตัวอย่าง และข้อความสรุปภาษาอังกฤษสำหรับบทความ
- ดาวน์โหลดผลเป็น CSV / JSON ได้

เบราว์เซอร์ตัดเหลือเฉพาะคอลัมน์ที่ใช้ (subject/video/frame, timestamp, lm11/12/23/24) แล้วบีบอัดก่อนส่ง ขนาดหลังบีบอัดต้องไม่เกิน 4.3 MB (ประมาณ frames.csv ดิบ 100 MB) เซิร์ฟเวอร์ประมวลผลในหน่วยความจำและไม่เก็บไฟล์ ปุ่ม "ลองกับข้อมูลจำลอง" ใช้ดูรูปแบบผลลัพธ์ได้โดยไม่ต้องมีไฟล์

| ส่วน | ไฟล์ |
|---|---|
| หน้าเว็บ | `public/index.html`, `public/app.js`, `public/style.css` |
| API วิเคราะห์ | `api/analyze.py` (POST ไฟล์, GET `?demo=1`) → `webapp/analysis.py` (ใช้ `decision.py` ตัวเดียวกับสคริปต์) |
| self-test | `api/selftest.py` |
| ทดสอบ | `python tests/test_webapp.py` |

การ train GRU และการดึง landmark ด้วย MediaPipe ยังต้องรันบนเครื่องของคุณ (TensorFlow ใหญ่เกินที่ Vercel รับได้) Vercel ติดตั้งแค่ `requirements.txt` (numpy, pandas) ส่วนสคริปต์ทั้งหมดใช้ `requirements-full.txt` push ขึ้น `main` แล้ว Vercel จะ deploy production ให้อัตโนมัติ

ติดตั้ง: `pip install -r requirements-full.txt` (โค้ดทดสอบแล้วกับ Python 3.13, TensorFlow 2.21 / Keras 3, MediaPipe 0.10.32)

---

## 1. เตรียมข้อมูล

### 1.1 `frames.csv` – หนึ่งแถวต่อหนึ่งเฟรม

| คอลัมน์ | ความหมาย |
|---|---|
| `subject_id` | รหัสอาสาสมัคร (ต้องตรงกันทุกวิดีโอของคนเดียวกัน) |
| `video_id` | รหัสวิดีโอ ไม่ซ้ำกัน **หนึ่งวิดีโอต่อหนึ่งการทดลอง** ถ้าคลิปยาวมีหลายการล้ม ให้ตัดเป็นหลาย `video_id` |
| `frame_idx` | ลำดับเฟรมในวิดีโอ (0, 1, 2, …) |
| `timestamp` | วินาที (ไม่มีก็ได้ จะคำนวณจาก `frame_idx / FPS`) |
| `label` | คลาสของเฟรม (ชื่อคลาสหรือเลข 0..K-1 ตามลำดับใน `CLASS_NAMES`) |
| `lm0_x, lm0_y, lm0_z, lm0_v, …, lm32_v` | landmark ของ MediaPipe 33 จุด (x, y, z, visibility) |

ถ้าเก็บ landmark เป็น array ต่อวิดีโอ แปลงได้ดังนี้

```python
import pandas as pd
parts = []
for subject_id, video_id, lms, labels in my_videos:        # lms: (T, 33, 4) จาก MediaPipe
    T = len(lms)
    df = pd.DataFrame(lms.reshape(T, -1), columns=[f"lm{i}_{a}" for i in range(33) for a in "xyzv"])
    df.insert(0, "label", labels)
    df.insert(0, "frame_idx", range(T))
    df.insert(0, "video_id", video_id)
    df.insert(0, "subject_id", subject_id)
    parts.append(df)
pd.concat(parts).to_csv("data/frames.csv", index=False)
```

### 1.2 `trials.csv` – หนึ่งแถวต่อหนึ่งการทดลอง (สำหรับ lead time / ablation)

| คอลัมน์ | ความหมาย |
|---|---|
| `video_id` | ตรงกับใน frames.csv |
| `trial_type` | `fall` หรือ `adl` |
| `scenario` | ชื่อสถานการณ์ เช่น `forward_fall`, `sit_fast` (ใช้ทำตารางแยกสถานการณ์) |
| `impact_frame` | **เฟรมที่ลำตัว/สะโพกกระทบเบาะครั้งแรก** (จำเป็นสำหรับ fall) |
| `onset_frame` | เฟรมเริ่มเสียสมดุล (ไม่บังคับ ถ้ามีจะรายงานเวลาจาก onset ถึงการเตือนด้วย) |

ถ้าวิดีโอยาวหนึ่งไฟล์มีหลายการทดลอง ใส่หลายแถวที่ `video_id` เดียวกันได้ โดยเพิ่มคอลัมน์ `trial_id` (ไม่ซ้ำกัน) และ `start_frame` / `end_frame` (ช่วงเฟรมของการทดลองนั้น)

**การกำหนด impact_frame:** นิยามให้ชัดในบทความ (เช่น "เฟรมแรกที่เชิงกราน/ลำตัวสัมผัสเบาะ") แล้วไล่ดูวิดีโอทีละเฟรม ถ้าเป็นไปได้ให้คน 2 คนกำหนดแยกกันแล้วรายงานความต่าง ความละเอียดของเวลาคือ ±1 เฟรม (±33 ms ที่ 30 fps) ซึ่งต้องระบุในบทความ ใช้ `lead_time.py --suggest-impact` ช่วยเสนอค่าเริ่มต้นได้ แต่**ต้องตรวจด้วยตาทุกค่า**

---

## 2. แก้ `config.py` ให้ตรงกับงาน (สำคัญที่สุด)

ทุกบรรทัดที่มี `>>> EDIT` ต้องตรวจ โดยเฉพาะ

1. `CLASS_NAMES` ต้องเรียงตามลำดับ output ของ softmax และ `POSITIVE_CLASSES` คือคลาสที่นับเป็น "สัญญาณเตือน"
2. `FPS`, `FRAME_W`, `FRAME_H`, `WINDOW`, `STANDARDIZE`
3. `model.py` ต้องมีสถาปัตยกรรม GRU และ optimizer **เหมือนในบทความ** ไม่เช่นนั้นผล LOSO จะเป็นของโมเดลอื่น (ถ้าใช้ PyTorch ต้องเขียน `model.py` และส่วน train/predict ใน `loso.py` ใหม่)
4. `hip_descent_velocity()` ใน `data.py` ต้องคำนวณ**เหมือนสมการ (3)–(4)** ทุกอย่าง (จุด landmark, lag, การ smooth, ค่าแปลงหน่วย)
5. `PERSIST_ON` ระบบจริงรวมสองกฎอย่างไร
   - `"gru"`: GRU เป็นบวก 3 เฟรมติดกัน และ ณ เฟรมนั้นความเร็ว ≥ 1.5
   - `"combined"`: ต้องผ่านทั้งสองเงื่อนไขพร้อมกัน 3 เฟรมติดกัน
6. `FRAME_H_METERS` ความสูงจริง (เมตร) ที่ภาพครอบคลุม ณ ระยะที่คนยืน ได้จากการคาลิเบรต เช่น วางไม้ยาว 1 m ที่ตำแหน่งคนยืน แล้ววัดจำนวน pixel

### ความเร็วแบบไม่ต้องคาลิเบรต (R1 ข้อ 3)

ตั้ง `VELOCITY_MODE = "torso"` ความเร็วจะถูกหารด้วยความยาวลำตัวในภาพ (ไหล่–สะโพก ค่ามัธยฐานย้อนหลังประมาณ 1 วินาที) ได้หน่วยเป็น **torso-lengths/s (TL/s)** ซึ่งไม่ขึ้นกับระยะกล้องและความสูงของคน ช่วงไหล่–สะโพกยาวประมาณ 0.29 × ส่วนสูง ดังนั้น 1.5 m/s ≈ 3 TL/s สำหรับคนสูง 1.65 m ให้ใช้ `ablation.py --loso-thresholds` เลือกค่า `V_THRESH` ใหม่ในหน่วยนี้ แทนการแปลงตรงๆ

---

## 3. ลำดับการรัน

```bash
# 0) ทดลองกับข้อมูลจำลองก่อน (CPU 2 core ใช้เวลาประมาณ 6-8 นาที)
python make_demo_data.py --out data_demo
python loso.py --frames data_demo/frames.csv --mode all --epochs 15 --out results_demo --models models_demo

# 1) LOSO + เทียบกับการแบ่งแบบเดิม
python loso.py --frames data/frames.csv --mode all

# 2) (เฉพาะเมื่อวิดีโอทดสอบ 150 สถานการณ์ / ADL 80 ครั้ง บันทึกแยกจากชุด train)
python predict_trials.py --frames data/trial_frames.csv --loso-dir models/loso \
       --model models/final.keras --out results/trial_predictions.csv
#    ถ้าวิดีโอทดสอบอยู่ในชุดเดียวกับ train ให้ใช้ results/loso/oof_predictions.csv แทนได้เลย

# 3) บน Raspberry Pi 5 / Jetson: วัด latency, FPS, RAM
python benchmark_edge.py --video sample.mp4 --model models/loso/S01.keras \
       --pose-model pose_landmarker_full.task --tag rpi5

# 4) Lead time
python lead_time.py --frames data/frames.csv --trials data/trials.csv --suggest-impact   # ช่วยกำหนด impact
python lead_time.py --frames data/frames.csv --trials data/trials.csv \
       --preds results/loso/oof_predictions.csv --latency-ms <p95 จากข้อ 3>

# 5) Ablation + ความไวของ threshold + เลือก threshold แบบ LOSO
python ablation.py --frames data/frames.csv --trials data/trials.csv \
       --preds results/loso/oof_predictions.csv --latency-ms <p95 จากข้อ 3> --loso-thresholds
```

---

## 4. ผลลัพธ์ที่ได้และนำไปใช้ในบทความ

| ไฟล์ผลลัพธ์ | ใช้ทำ |
|---|---|
| `results/split_comparison.md` | ตาราง: random 80/20 vs video-independent vs LOSO (accuracy, macro-F1 แบบ mean ± SD และ pooled) |
| `results/loso/fold_metrics.csv`, `confusion_matrix.png` | ผลรายบุคคล และ confusion matrix ของ LOSO |
| `results/lead_time/summary.md`, `lead_time_hist.png` | ตาราง/กราฟ lead time, สัดส่วนที่เตือนได้ ≥ 100 ms |
| `results/lead_time/timelines/*.png` | รูปตัวอย่างไทม์ไลน์ (ความสูงสะโพก, ความเร็ว, ความน่าจะเป็น GRU, จุดเตือน, จุดกระทบ) |
| `results/ablation/ablation_table.md` | ตาราง ablation พร้อม McNemar p |
| `results/ablation/sensitivity_heatmaps.png`, `tradeoff.png` | แสดงว่า threshold ที่เลือกอยู่ในช่วงที่ผลคงที่ |
| `results/ablation/loso_thresholds.md` | ผลเมื่อเลือก threshold โดยไม่เห็นผู้ทดลองที่ทดสอบ |
| `results/benchmark/<tag>_summary.md` | ตาราง latency/FPS/RAM บนอุปกรณ์ edge |

### ตัวอย่างประโยค (แทน XX ด้วยผลจริง)

> To rule out temporal and subject leakage, the GRU was re-evaluated with leave-one-subject-out cross-validation (XX folds). Windows were built within each recording only, and early stopping used validation subjects drawn from the training fold. LOSO accuracy was XX.X ± X.X % (macro-F1 XX.X ± X.X %), compared with 97.03 % under the original random split (Table X).

> Impact was defined as the first frame in which the pelvis/trunk contacted the mat, annotated frame by frame (resolution ±33 ms at 30 fps). After subtracting the p95 processing latency measured on a Raspberry Pi 5 (XX ms), XX of XX falls (XX %) were warned before impact, with a median effective lead time of XXX ms (IQR XXX–XXX ms).

> With the same GRU outputs, the GRU alone produced false alarms in XX/80 high-acceleration ADL trials; adding the three-frame persistence rule and the velocity threshold reduced this to 4/80 (exact McNemar p = X.XXX), at a cost of XX ms in median lead time (Table X).

---

## 5. ข้อควรระวัง

- **อย่าเลือก threshold จากข้อมูลชุดเดียวกับที่ใช้รายงานผล** ให้ระบุว่า threshold ถูกกำหนดไว้ล่วงหน้า แล้วแสดง heatmap ประกอบ หรือรายงานผลจาก `--loso-thresholds`
- ถ้าค่า LOSO ต่ำกว่า 97.03 % ให้รายงานตามจริงและอธิบายสาเหตุ ผู้ทรงคุณวุฒิทั้ง 3 ท่านต้องการเห็นตัวเลขนี้
- **False alarms / h** มีความหมายเฉพาะเมื่อมีการบันทึก ADL ต่อเนื่องนานๆ สำหรับการทดลองสั้นๆ ให้รายงานเป็นจำนวนครั้ง (เช่น 4/80)
- Lead time ที่คำนวณจากเฟรมยังไม่รวมเวลาค้างของกล้อง (capture latency) และเวลาส่ง LINE วัดเวลาส่ง LINE แยกได้แบบนี้ แล้วรายงานค่า median/p95 จาก 20–50 ครั้ง

  ```python
  import time
  t0 = time.perf_counter(); send_line_alert(...); print(f"{(time.perf_counter() - t0) * 1000:.0f} ms")
  ```
- ถ้าระบบจริงเรียก `model.predict()` ทุกเฟรม จะช้ากว่า `tf.function` มาก (ในเครื่องทดสอบ ~65 ms เทียบกับ ~2 ms ต่อเฟรม) ซึ่งเท่ากับเสีย lead time ไปตรงๆ ลองรัน `benchmark_edge.py --keras-predict` เทียบกันดู
- MediaPipe ตั้งแต่ 0.10.30 ไม่มี `mp.solutions` แล้ว ต้องใช้ `--pose-model pose_landmarker_*.task` (Tasks API) และตั้งค่าให้ตรงกับระบบจริง (lite/full/heavy มีผลต่อ FPS มาก)

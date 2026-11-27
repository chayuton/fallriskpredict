# ชุดข้อมูล GMDCSA24 พร้อมอัปโหลดเข้า fallriskpredict

เตรียมเมื่อ 27 พ.ย. 2026 จากวิดีโอต้นฉบับของ GMDCSA24 (GitHub ekramalam, commit 5abac76 = Zenodo v2.1) ด้วยสคริปต์ใน `scripts/` (ใน repo [fallriskpredict](https://github.com/chayuton/fallriskpredict) อยู่ที่ `datasets/gmdcsa24/`) ซึ่งรันซ้ำได้ทุกขั้น (ข้อ 8)

## 1. อัปโหลดเข้าเว็บแอป (https://fallriskpredict.vercel.app)

| ช่องในเว็บแอป | ไฟล์ |
|---|---|
| frames | `upload/frames.csv` |
| trials | `upload/trials.csv` |
| predictions | `upload/predictions.csv` (คลาส loss_of_balance ถูกเลือกเป็นสัญญาณเตือนให้อัตโนมัติ) |

ค่าที่ตั้ง: Velocity mode = **torso (TL/s)**, V = **3.0** TL/s (เทียบเท่าประมาณ 1.5 m/s เมื่อลำตัวยาวราว 0.5 m), P = 0.5, N = 3, persist on = gru, fps = **30**, Latency = ค่า p95 ที่วัดด้วย `benchmark_edge.py` บนอุปกรณ์จริง

ใช้โหมด torso เพราะชุดนี้ไม่มีข้อมูลคาลิเบรตกล้อง โหมด meters จึงแปลงเป็น m/s ไม่ได้จริง

## 2. ไฟล์ในชุดนี้

| ไฟล์ | เนื้อหา |
|---|---|
| `upload/frames.csv` | 38,514 เฟรม (160 วิดีโอ, 4 คน) ที่ 30 fps เฉพาะคอลัมน์ที่เว็บแอปใช้ (ไหล่ 11–12, สะโพก 23–24) พร้อม label |
| `upload/trials.csv` | 150 trial: ล้ม 69, ADL 81 พร้อม onset_frame / impact_frame / impact_source |
| `upload/predictions.csv` | ความน่าจะเป็นจาก GRU อ้างอิงแบบ LOSO (out-of-fold) prob_normal, prob_loss_of_balance, prob_fallen |
| `full_landmarks/frames_30fps_all33.csv` | landmark ครบ 33 จุด × (x, y, z, visibility) สำหรับ train หรือรันโมเดลของคุณเอง |
| `labels/` | manifest.csv, segments.csv (label เดิมของชุดข้อมูล), segments_split.csv (แบ่งช่วงล้มที่จุดกระแทก), trials_suggested.csv (ค่าเสนอดิบจาก toolkit), impact_review.csv (ค่าประมาณทุกวิธี, หมายเหตุ, ภาพที่ใช้ตรวจ) |
| `impact_check/` | impact_check_XX.jpg (5 เฟรมรอบจุดกระแทกของทุกคลิปล้ม), strip_XX.jpg (เฟรมทุก 2 เฟรมของทุกคลิปที่อยู่ใน trials กรอบแดง = impact ที่ใช้, H/T/K = ค่าประมาณจากสะโพก/ลำตัว/เข่า) |
| `reference_results/` | ผลวิเคราะห์ด้วย engine เดียวกับเว็บแอป และผล LOSO ของ GRU อ้างอิง |
| `scripts/` | gmdcsa24_labels.py (อ่าน label), build_gmdcsa24_bundle.py (ขั้นตอนทั้งหมด) ต้องรันจากโฟลเดอร์ของ repo fallriskpredict |
| `GET_MORE_DATA_TH.md`, `get_public_data.py` | วิธีโหลดและแปลงชุดข้อมูลเปิดอื่น (URFD, OmniFall ฯลฯ) |

## 3. ขั้นตอนที่ทำ

1. **Label** อ่านคอลัมน์ Classes ของ `Subject N/{Fall,ADL}.csv` เป็นช่วงเวลา (338 ช่วง) แก้ 2 คลิปที่ label ผิดรูปแบบ (S4/Fall/15 ไม่มีช่วงเวลา, S2/Fall/10 มีช่วงล้มต่อกัน 2 ช่วง)
2. **Pose** MediaPipe Pose 0.10.14 (legacy API, model_complexity = 1) ทุกเฟรม เก็บ 33 จุด × (x, y, z, visibility) และคอลัมน์ detected (พบคนใน 96.9% ของเฟรม)
3. **30 fps** วิดีโอส่วนใหญ่ 29.3–29.9 fps, 29 คลิปเป็น 15 fps จึง interpolate เป็น 30 fps (ไม่เติมช่วงที่ไม่พบคนนานกว่า 0.25 s)
4. **Impact** ชุดนี้มีเวลาเริ่มล้ม (onset) แต่ไม่มีเวลากระแทก กำหนดนิยามว่า impact = **เฟรมแรกที่มือ เข่า สะโพก หรือหลังแตะพื้น (หรือเตียง)** ประมาณอัตโนมัติจากจุดสิ้นสุดการลดระดับช่วงแรกของสะโพก ลำตัว และเข่า แล้ว**เปิดดูภาพทุกคลิปที่อยู่ใน trials**: แก้ 31 คลิป (หมายเหตุอยู่ใน impact_source) อีก 38 คลิปค่าอัตโนมัติใกล้เคียงภาพ (ประมาณ ±0.1 s)
5. **Label รายเฟรม** normal / loss_of_balance = [onset, impact) / fallen = [impact, จบวิดีโอ]
6. **Trials** หนึ่ง trial ต่อวิดีโอ ตัด 10 คลิปล้มออกจาก trials (แต่ยังใช้ train): 7 คลิปเริ่มล้มภายใน 0.5 s แรก (GRU ยังไม่มีประวัติ 15 เฟรม) และ 3 คลิปไถลลงจากเตียงช้าๆ ที่ไม่มีจุดกระแทกเดียว (S2/Fall/15, S4/Fall/06, S4/Fall/14)
7. **Predictions** GRU อ้างอิงของ toolkit (ค่าเริ่มต้นใน config.py: หน้าต่าง 15 เฟรม, 132 feature, class weight, early stopping) แบบ leave-one-subject-out 4 fold

## 4. ผลอ้างอิง (GRU อ้างอิง ไม่ใช่โมเดลในบทความ)

LOSO ระดับหน้าต่าง: accuracy 83.9 ± 5.9%, macro-F1 64.0 ± 5.8%, sensitivity ของคลาสเตือน 38.4 ± 7.3% (เฉลี่ย ± SD ของ 4 fold)

ผลจากเว็บแอปด้วยค่าตั้งในข้อ 1 (latency = 0):

| การตั้งค่า | ตรวจพบการล้ม | เตือนก่อนกระแทก | Lead time มัธยฐาน [IQR] | ADL ที่แจ้งผิด | แจ้งผิด/ชม. |
|---|---|---|---|---|---|
| GRU อย่างเดียว (N=1) | 65/69 | 52/69 | 200 [67, 300] ms | 47/81 | 270.5 |
| GRU + persistence (N=3) | 64/69 | 46/69 | 133 [0, 208] ms | 46/81 | 265.7 |
| GRU + velocity (N=1) | 51/69 | 36/69 | 100 [−17, 200] ms | 15/81 | 72.5 |
| GRU + persistence + velocity (ที่เสนอ) | 49/69 | 32/69 | 67 [−33, 167] ms | 14/81 | 67.6 |
| Velocity อย่างเดียว (N=1) | 61/69 | 46/69 | 167 [33, 300] ms | 28/81 | 144.9 |

ตัวเลขนี้ใช้ตรวจว่าไฟล์และขั้นตอนทำงานครบ ไม่ใช่ผลที่ควรใส่ในบทความ ให้แทน `predictions.csv` ด้วยผลจากโมเดลของคุณ (ข้อ 6)

## 5. ข้อควรระวังเมื่อใช้ในรายงาน

- เป็นการล้มจำลองโดยคนอายุน้อย 4 คนในบ้าน 3 หลัง หลายคลิปค่อยๆ ย่อลงคุกเข่าหรือเอามือยันพื้นก่อน ระยะ onset → impact มัธยฐานเพียง 0.43 s (onset ของชุดข้อมูลบางคลิปช้ากว่าการเคลื่อนไหวจริง) lead time จากชุดนี้จึงไม่แทนการล้มจริงของผู้สูงอายุ
- impact อ่านจากภาพ 30 fps ด้วยตา คลาดเคลื่อนได้ราว ±0.1 s และยังต้องให้ผู้วิจัยตรวจซ้ำก่อนตีพิมพ์ (ใช้ `impact_check/strip_XX.jpg` และแก้ `impact_frame` ใน trials.csv ได้โดยตรง) ระบุวิธีกำหนด impact ในบทความ
- ADL ใน trials รวมเพียง 12.4 นาที ค่าแจ้งผิดต่อชั่วโมงจึงแกว่งมาก ให้รายงานจำนวนครั้งคู่กันเสมอ
- MediaPipe มักหาคนไม่เจอตอนนอนกับพื้นหรือในที่มืด (เช่น S1/Fall/02 พบคน 40% ของเฟรม) เฟรมที่ไม่พบคนมีค่า landmark = 0 และ detected = 0
- มีเพียง 4 คน LOSO จึงมี 4 fold ค่าความแปรปรวนระหว่าง fold สูง เหมาะเป็นการตรวจสอบภายนอกเสริมชุดข้อมูลหลักของคุณ

## 6. ใช้กับโมเดลของคุณ

```
python predict_trials.py --frames full_landmarks/frames_30fps_all33.csv --model my_gru.keras --out my_predictions.csv
```

ถ้าโมเดลของคุณ train ด้วยข้อมูลอื่น (ไม่เคยเห็น 4 คนนี้) ผลนี้คือการทดสอบภายนอกโดยตรง ตรวจให้ลำดับ feature และคลาสตรงกับตอน train (FEATURE_COLS, CLASS_NAMES ใน config.py) แล้วอัปโหลด my_predictions.csv แทน `upload/predictions.csv`

หรือ train GRU ใหม่ด้วย LOSO: `python loso.py --frames full_landmarks/frames_30fps_all33.csv --mode all --out results_gmdcsa24`

## 7. แหล่งที่มาและสิทธิ์การใช้

ข้อมูลในชุดนี้ (landmark, label, ภาพตัวอย่าง) สร้างจาก GMDCSA24 ซึ่งเผยแพร่ภายใต้ MIT License (ดู `LICENSE_GMDCSA24_MIT.txt`) เผยแพร่ต่อได้โดยคงประกาศสิทธิ์และอ้างอิงแหล่งที่มา

- Alam E., Sufian A., Dutta P., Leo M., Hameed I.A. GMDCSA-24: A dataset for human fall detection in videos. *Data in Brief* 57 (2024) 110892. https://doi.org/10.1016/j.dib.2024.110892
- ข้อมูล: https://doi.org/10.5281/zenodo.13354453

## 8. สร้างชุดนี้ใหม่

รันจากโฟลเดอร์ของ repo fallriskpredict (ต้องมี mediapipe, opencv-python, tensorflow ตาม requirements-full.txt)

```
git clone https://github.com/ekramalam/gmdcsa24-a-dataset-for-human-fall-detection-in-videos data/raw/gmdcsa24
python datasets/gmdcsa24/gmdcsa24_labels.py data/raw/gmdcsa24 data/gmdcsa24
python prepare_open_data.py extract --dir data/gmdcsa24
python datasets/gmdcsa24/build_gmdcsa24_bundle.py label
python datasets/gmdcsa24/build_gmdcsa24_bundle.py train
python datasets/gmdcsa24/build_gmdcsa24_bundle.py verify
python datasets/gmdcsa24/build_gmdcsa24_bundle.py package --videos data/raw/gmdcsa24
```

ผลอยู่ที่ `results/gmdcsa24_upload_bundle/` และ `results/gmdcsa24_upload_bundle.zip` ขั้น extract ใช้เวลานานที่สุด (ราว 15 นาทีบน CPU) และทำต่อจากวิดีโอที่ค้างได้ ถ้าใช้ MediaPipe รุ่นที่ไม่มี `mp.solutions` แล้ว ให้เพิ่ม `--pose-model pose_landmarker_full.task` ซึ่ง landmark อาจต่างจากชุดนี้เล็กน้อย

แก้จุดกระแทกหลังตรวจภาพ: แก้ตาราง `MANUAL_IMPACT_S` (เวลาเป็นวินาที) หรือ `MANUAL_EXCLUDE` ใน `build_gmdcsa24_bundle.py` แล้วสั่ง `label --skip-prep`, `train`, `verify`, `package` ใหม่ เพราะ label รายเฟรมเปลี่ยนตาม impact

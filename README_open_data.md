# ชุดข้อมูลเปิดที่ใช้กับสคริปต์ชุดนี้ได้

ตรวจสอบข้อมูลเมื่อ 26 พ.ย. 2026 สิทธิ์การใช้และวิธีดาวน์โหลดอาจเปลี่ยนได้ ให้ตรวจที่หน้าเว็บต้นทางอีกครั้งก่อนใช้

## 1. เลือกชุดข้อมูลตามความเห็นผู้ทรงคุณวุฒิ

| ต้องการตอบ | ชุดที่แนะนำ | เหตุผล |
|---|---|---|
| LOSO / ทดสอบกับคนที่ไม่เคยเห็น (R1 ข้อ 1, R2 ข้อ 1, R3) | **UP-Fall** (17 คน), **CMDFall** (50 คน), **CAUCAFall** (10 คน) ผ่าน label ของ **OmniFall** | มีรหัสผู้ทดลอง และ label รูปแบบเดียวกันทุกชุด |
| Lead time (R1 ข้อ 2, R2 ข้อ 2) | ชุดใดก็ได้ที่มีช่วง "fall" (OmniFall) หรือ **URFD** (label 0 = กำลังล้ม) | ได้เฟรมเริ่มล้ม (onset) จาก label ส่วนเฟรมกระแทกต้องกำหนดเอง (มีตัวช่วยเสนอค่า) |
| False alarm กับกิจกรรมที่เร็ว / ablation (R2 ข้อ 5) | UP-Fall (กระโดด, หยิบของ), CMDFall (กระโดดอยู่กับที่, เซ, วิ่งช้า, ย่อตัว), CAUCAFall (กระโดด, คุกเข่า) | มี ADL ที่มีความเร่งสูงหลายแบบ |
| กลุ่มผู้สูงอายุ (R2 ข้อ 4) | **SFU real-life falls** (การล้มจริงของผู้สูงอายุ), **ETRI-Activity3D** (ADL ผู้สูงอายุเทียบคนหนุ่มสาว), **Toyota Smarthome Untrimmed** (ADL ต่อเนื่องของผู้สูงอายุ) | ไม่มีชุดข้อมูลเปิดที่ผู้สูงอายุจำลองการล้มพร้อม label เวลากระแทก ชุดเหล่านี้ใกล้เคียงที่สุด |

## 2. รายละเอียดแต่ละชุด

| ชุดข้อมูล | ผู้ทดลอง (อายุ) | วิดีโอ | ล้ม / ADL | Label ที่ใช้ได้ | การเข้าถึง / สิทธิ์ |
|---|---|---|---|---|---|
| **OmniFall** (label อย่างเดียว) | รวม 8 ชุด staged + OOPS-Fall + ชุดสังเคราะห์ OF-Syn | – | 16 คลาส: walk, fall, fallen, sit_down, lie_down, jump … | ช่วงเวลา (วินาที) + subject + camera, มี split แบบ cross-subject | Hugging Face `simplexsigil2/omnifall`, CC BY-NC(-SA) 4.0, วิดีโอต้องโหลดจากชุดต้นทาง |
| **UP-Fall** | 17 คน (18–24 ปี) | 2 กล้อง, 18 fps, 640×480 | ล้ม 5 แบบ, ADL 6 แบบ, อย่างละ 3 ครั้ง | ผ่าน OmniFall | เว็บไซต์ HAR-UP (Google Sites) ทั้งชุดใหญ่มาก ให้โหลดเฉพาะข้อมูลกล้อง |
| **CMDFall** | 50 คน (21–40 ปี) | 7 Kinect, 20 fps, 640×480 | ล้ม 8 แบบ, ADL 12 แบบ ในวิดีโอต่อเนื่อง | ผ่าน OmniFall (ใช้ `--mode events`) | อีเมลขอผู้จัดทำ (MICA), ฟรีเพื่อการวิจัย |
| **CAUCAFall** | 10 คน (หลายช่วงอายุ ไม่ระบุช่วง) | 1 กล้อง IR, 23 fps | ล้ม 5 แบบ, ADL 5 แบบ | label ราย frame (fall/nofall) + OmniFall | Mendeley Data, CC BY 4.0 |
| **GMDCSA24** | 4 คน | webcam 30 fps, 720p | ล้ม 81, ADL 79 คลิป | CSV เวลาเริ่ม/จบ + OmniFall | Zenodo 10.5281/zenodo.13354453, CC BY 4.0 |
| **Le2i (ImViA)** | 9 คน | 25 fps, 320×240 | ล้ม 143, ADL 79 | เฟรมเริ่ม/จบการล้ม + OmniFall | search-data.ubfc.fr (ลิงก์จากหน้า OmniFall), เหมาะเป็นชุดทดสอบภายนอก |
| **URFD** | 5 คน | Kinect 30 fps, 640×480 | ล้ม 30, ADL 40 | ราย frame: −1 ไม่นอน, 0 กำลังล้ม, 1 นอนกับพื้น | ดาวน์โหลดตรง, CC BY-NC-SA 4.0, **ไม่มีรหัสผู้ทดลองราย sequence** → ใช้ทดสอบภายนอกเท่านั้น |
| **SAFER-Activities** (2026) | 46 คน (18–58 ปี) | หลายกล้อง | การล้ม 5,406 ครั้ง, 30 คลาส รวมผู้ใช้วีลแชร์ | ราย frame, split ตามผู้ทดลอง | Hugging Face, CC BY-NC-SA 4.0, ต้องเขียน `segments.csv` เอง |
| **SFU real-life falls** | ผู้สูงอายุในสถานดูแล 14 คน (เฉลี่ย 84 ปี) ในชุดสาธารณะ | กล้องวงจรปิด หลายมุมรวมในไฟล์เดียว | การล้มจริง 28 คลิป (ชุดเต็ม 300 คลิปต้องขอสิทธิ์) | ไม่มี label เวลา ต้องกำหนด onset/impact เอง | Databrary volume 739 (ติดต่อ Steve Robinovitch, SFU) |
| **ETRI-Activity3D** | ผู้สูงอายุ 50 คน (64–88 ปี) + คนอายุ 20 ปี 50 คน | Kinect v2 RGB-D | ADL 55 แบบ (ไม่มีการล้ม) | label ราย sample | พอร์ทัล ETRI |
| **Toyota Smarthome Untrimmed** | ผู้สูงอายุ 18 คน (60–80 ปี) | 7 กล้อง, 640×480, วิดีโอยาวเฉลี่ย 21 นาที | ADL อย่างเดียว | ช่วงกิจกรรม | ขอสิทธิ์ใช้งานผ่านเว็บไซต์โครงการ (Inria) |
| **MUVIM** | ผู้ใหญ่ 30 คน (ล้ม + ADL), ผู้สูงอายุ 10 คน (ADL) | RGB, thermal, depth | – | เฟรมเริ่ม/จบการล้ม | อีเมลขอ + เอกสารยินยอมเรื่องความเป็นส่วนตัว |
| **KFall** | 32 คน | IMU อย่างเดียว (ไม่มีวิดีโอ) | ล้ม 15, ADL 21 แบบ | onset / impact จากวิดีโอที่ซิงก์ | ใช้อ้างอิงนิยาม onset/impact และค่า lead time เปรียบเทียบ |

นิยาม label ของ OmniFall: ช่วง `fall` เริ่มที่เฟรมแรกที่ท่าทางเริ่มต่างจากกิจกรรมก่อนหน้า และ**จบเมื่อการเคลื่อนที่จากแรงเฉื่อยหยุด ไม่ใช่ตอนกระทบพื้น** จึงใช้ต้นช่วงเป็น onset ได้ แต่ต้องหาเฟรมกระแทกเพิ่ม

## 3. แปลงข้อมูลด้วย `prepare_open_data.py`

### ตัวอย่าง: UP-Fall ผ่าน OmniFall

```python
# 1) โหลด label ของ OmniFall (ไฟล์เล็ก)
from huggingface_hub import snapshot_download
snapshot_download("simplexsigil2/omnifall", repo_type="dataset",
                  allow_patterns=["labels/*", "splits/*"], local_dir="omnifall")
```

```bash
# 2) จับคู่ label กับวิดีโอ (วางไฟล์ตามโครงสร้าง path ในไฟล์ label: .mp4/.avi/.zip หรือโฟลเดอร์รูปก็ได้)
python prepare_open_data.py omnifall --labels omnifall/labels/up_fall.csv \
       --video-root up_fall=/data/UP-Fall --out data/upfall
#    ดู data/upfall/manifest.csv: แถวที่คอลัมน์ media ว่างคือหาไฟล์ไม่เจอ แก้ path เองได้

# 3) ดึง landmark ด้วย MediaPipe (หยุดกลางคันแล้วสั่งใหม่ได้ จะทำต่อจากวิดีโอที่ค้าง)
python prepare_open_data.py extract --dir data/upfall --pose-model pose_landmarker_full.task

# 4) ปรับเป็น 30 fps ให้ตรงกับที่ GRU ถูก train (หน้าต่าง 15 เฟรม)
python prepare_open_data.py resample --frames data/upfall/frames.csv --fps 30

# 5) สร้าง trials.csv (onset จาก label, impact เสนอจากวิถีสะโพก -> ต้องตรวจด้วยตา)
python prepare_open_data.py trials --dir data/upfall --frames data/upfall/frames_30fps.csv --mode video
python lead_time.py --frames data/upfall/frames_30fps.csv --trials data/upfall/trials.csv --suggest-impact

# 6) ใช้กับสคริปต์เดิมได้ทันที
python loso.py --frames data/upfall/frames_30fps.csv --mode loso
python lead_time.py --frames data/upfall/frames_30fps.csv --trials data/upfall/trials.csv \
       --preds results/loso/oof_predictions.csv
python ablation.py  --frames data/upfall/frames_30fps.csv --trials data/upfall/trials.csv \
       --preds results/loso/oof_predictions.csv --loso-thresholds
```

### ชุดอื่น

- **CMDFall / MCFD** (วิดีโอยาว มีหลายเหตุการณ์): ใช้ `trials --mode events` จะได้หนึ่ง trial ต่อการล้มหรือ ADL หนึ่งครั้ง (มีคอลัมน์ `start_frame`/`end_frame`)
- **URFD**: `python prepare_open_data.py urfd --falls urfall-cam0-falls.csv --adls urfall-cam0-adls.csv --video-root /data/URFD --out data/urfd` แล้ว `extract`, `trials --impact segment_end` (เฟรมสุดท้ายที่ label = "กำลังล้ม" ใช้แทนจุดกระทบได้โดยประมาณ)
- **SFU real-life falls**: เขียน `manifest.csv` เอง (video_id, subject_id, media, crop) ใช้คอลัมน์ `crop = "x,y,w,h"` ตัดเอาเฉพาะมุมกล้องเดียว แล้วกำหนด onset/impact ใน trials.csv ด้วยตา
- **ชุดอื่นๆ**: เขียน `segments.csv` (video_id, subject_id, label, start_s, end_s) และ `manifest.csv` แล้วเริ่มที่ขั้น `extract`
- **เปลี่ยนการแมป label** ไปเป็นคลาสของโมเดลคุณ: `extract --merge-only --label-map '{"fall": "loss_of_balance", "fallen": "fallen"}'` (คลาสอื่นทั้งหมดเป็น `--other-label`, ค่าเริ่มต้นคือ `normal`)

### ค่าที่ควรตั้งใน `config.py` เมื่อใช้ข้อมูลเปิด

- `VELOCITY_MODE = "torso"` เพราะชุดข้อมูลเปิดไม่มีข้อมูลคาลิเบรตกล้อง แล้วเลือก `V_THRESH` ใหม่ด้วย `ablation.py --loso-thresholds`
- `FPS = 30` ถ้าใช้ไฟล์ที่ผ่าน `resample` แล้ว
- `CLASS_NAMES` / `POSITIVE_CLASSES` ให้ตรงกับ `--label-map`
- ความละเอียดภาพแต่ละชุดต่างกันได้ (สคริปต์อ่านคอลัมน์ `frame_w`/`frame_h` ของแต่ละแถว)

## 4. ข้อควรระวังในการรายงาน

- หลายชุดใช้ได้เฉพาะงานวิจัยไม่ใช่เชิงพาณิชย์ (NC) และต้องอ้างอิงทั้งบทความ OmniFall และบทความของชุดย่อยทุกชุดที่ใช้
- เฟรมกระทบพื้นจาก `--impact suggest` เป็นค่าเสนอเท่านั้น ต้องตรวจทุกคลิปก่อนรายงาน lead time และระบุวิธีกำหนดในบทความ (อ้างนิยามของ KFall: onset = เริ่มล้ม, impact = ลำตัวกระทบพื้น)
- อัตราเฟรมของชุดข้อมูลต่างกัน (18–30 fps) ความละเอียดของ lead time จึงต่างกันด้วย ให้รายงานแยกตามชุดข้อมูล
- ทุกชุด (ยกเว้น SFU) เป็นการล้มจำลองโดยคนอายุน้อย ผลจากชุดเหล่านี้ช่วยตอบเรื่อง LOSO และการทดสอบข้ามสภาพแวดล้อม แต่**ยังไม่ได้แก้ข้อจำกัดเรื่องผู้สูงอายุ** ให้ใช้ SFU real-life falls (เชิงคุณภาพ/กลุ่มเล็ก) และ ETRI-Activity3D (เทียบ false alarm ระหว่างผู้สูงอายุกับคนหนุ่มสาว) แล้วเขียนเป็น limitation ให้ชัด
- OF-Syn (วิดีโอสังเคราะห์ที่มีกลุ่มอายุ 65+) ช่วยเรื่องรูปลักษณ์ของผู้สูงอายุ แต่การเคลื่อนไหวไม่ได้มาจากฟิสิกส์จริง ห้ามใช้ประเมิน lead time หรือความเร็ว

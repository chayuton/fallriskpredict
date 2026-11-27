# ชุดข้อมูลเปิดอื่นที่ต้องดาวน์โหลดบนเครื่องคุณ

ระบบที่ใช้เตรียมชุดนี้ออกอินเทอร์เน็ตได้เฉพาะ GitHub จึงเตรียมให้ครบทุกขั้นได้เฉพาะ GMDCSA24 (โฟลเดอร์ `upload/`) ชุดอื่นต้องโหลดบนเครื่องคุณ แล้วแปลงด้วยสคริปต์ของ repo [fallriskpredict](https://github.com/chayuton/fallriskpredict) ขั้นตอนเดียวกับที่ใช้สร้าง GMDCSA24

ตรวจสอบแหล่งข้อมูลเมื่อ 27 พ.ย. 2026 สิทธิ์และวิธีดาวน์โหลดอาจเปลี่ยนได้

## 1. โหลดอัตโนมัติได้ (ไม่ต้องขอสิทธิ์)

```
pip install huggingface_hub
python get_public_data.py            # URFD + OmniFall labels ลงโฟลเดอร์ public_data/
```

| ชุด | ได้อะไร | ใช้ตอบผู้ประเมิน | สิทธิ์ |
|---|---|---|---|
| URFD | วิดีโอกล้อง 0: ล้ม 30 + ADL 40 และ label รายเฟรม (−1 ปกติ, 0 กำลังล้ม, 1 นอนกับพื้น) | lead time (onset และจุดสิ้นสุดการล้มมาจาก label), ชุดทดสอบภายนอก | CC BY-NC-SA 4.0 (ไม่ใช่เชิงพาณิชย์) |
| OmniFall (label อย่างเดียว) | label รูปแบบเดียวกันของชุด staged 8 ชุด (UP-Fall, CMDFall, Le2i, CAUCAFall, GMDCSA24, EDF, OCCU, MCFD) พร้อม split แบบ cross-subject | LOSO หลายชุด, false alarm ของ ADL ที่เร็ว | CC BY-NC(-SA) 4.0 |

URFD ไม่มีรหัสผู้ทดลองราย sequence จึงใช้ LOSO ไม่ได้ ให้ใช้เป็นชุดทดสอบภายนอก (train ด้วยชุดอื่นแล้วใช้ `predict_trials.py --model`)

## 2. ต้องโหลดวิดีโอจากเว็บไซต์หรือขอสิทธิ์

| ชุด | ผู้ทดลอง | ใช้ตอบ | วิธีได้มา |
|---|---|---|---|
| UP-Fall | 17 คน | LOSO, ADL เร็ว (กระโดด หยิบของ) | ดาวน์โหลดเฉพาะวิดีโอกล้องจากเว็บไซต์ HAR-UP แล้วใช้ label ของ OmniFall |
| CAUCAFall | 10 คน | LOSO, กล้อง IR | Mendeley Data (CC BY 4.0) |
| Le2i | 9 คน | ชุดทดสอบภายนอก | ImViA / UBFC (ลิงก์จากหน้า OmniFall) |
| CMDFall | 50 คน | LOSO ขนาดใหญ่ | อีเมลขอผู้จัดทำ (MICA) วิดีโอยาว ใช้ `trials --mode events` |
| SFU real-life falls | ผู้สูงอายุในสถานดูแล | การล้มจริงของผู้สูงอายุ (เชิงคุณภาพ) | Databrary volume 739 (ขอสิทธิ์) ต้องกำหนด onset/impact เอง |
| ETRI-Activity3D, Toyota Smarthome | ผู้สูงอายุ (ADL อย่างเดียว) | อัตราแจ้งเตือนผิดในผู้สูงอายุ | ขอสิทธิ์ผ่านเว็บไซต์โครงการ |

ไม่ได้ใช้ไฟล์ skeleton ของ UP-Fall ที่มีผู้แปลงไว้บน GitHub เพราะหัวคอลัมน์และชื่อไฟล์ไม่สม่ำเสมอ และตำแหน่งสะโพกกระโดดเป็นช่วงๆ ซึ่งทำให้ค่าความเร็วสะโพกผิด ให้ดึง landmark จากวิดีโอต้นฉบับด้วย `extract` เพื่อให้ทุกชุดผ่านขั้นตอนเดียวกัน

## 3. แปลงเป็นไฟล์สำหรับอัปโหลด (ทุกชุดเหมือนกัน)

```
python prepare_open_data.py omnifall --labels <label ของชุดนั้น>.csv --video-root up_fall=/path/to/videos --out data/upfall
python prepare_open_data.py extract  --dir data/upfall
python prepare_open_data.py resample --frames data/upfall/frames.csv --fps 30
python prepare_open_data.py trials   --dir data/upfall --frames data/upfall/frames_30fps.csv --mode video --impact suggest
python lead_time.py --frames data/upfall/frames_30fps.csv --trials data/upfall/trials.csv --suggest-impact
python loso.py --frames data/upfall/frames_30fps.csv --mode loso --out results_upfall
```

จากนั้นอัปโหลด `frames_30fps.csv`, `trials.csv` และ `results_upfall/loso/oof_predictions.csv` เข้าเว็บแอป

สิ่งที่เรียนรู้จากการเตรียม GMDCSA24 และควรทำกับชุดอื่นด้วย:

- ค่า impact อัตโนมัติมักช้ากว่าจุดที่มือหรือเข่าแตะพื้นในการล้มแบบช้าๆ (ใน GMDCSA24 แก้ 31 จาก 69 คลิป) ต้องเปิดดูภาพทุกคลิปก่อนรายงาน lead time
- คลิปที่เริ่มล้มภายใน 0.5 วินาทีแรกควรตัดออกจาก trials เพราะ GRU ยังไม่มีประวัติ 15 เฟรม
- การไถลลงจากเตียงช้าๆ ไม่มีจุดกระแทกเดียว ควรตัดออกจากการวัด lead time
- อ้างอิงทั้งบทความ OmniFall และบทความของชุดย่อยทุกชุดที่ใช้

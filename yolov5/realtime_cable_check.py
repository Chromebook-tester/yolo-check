#!/usr/bin/env python3
"""
실시간 케이블 체크 시스템 (리팩토링 완료)
- 정상 케이블: 초록색 박스
- 비정상 케이블: 빨간색 박스
"""

import cv2
import torch
from pathlib import Path
import sys

# YOLOv5 경로 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class RealTimeCableChecker:
    def __init__(self, weights_path="runs/train/cable_check/weights/best.pt"):
        print("🔌 실시간 케이블 체커 시작!")
        try:
            self.model = torch.hub.load(str(ROOT), 'custom', path=weights_path, source='local', force_reload=True)
            print("✅ AI 모델 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            sys.exit(1)
        
        self.class_names = {
            0: "전원정상", 1: "USB정상", 2: "HDMI정상",
            3: "이더넷정상", 4: "오디오정상"
        }
        self.normal_color = (0, 255, 0)
        self.abnormal_color = (0, 0, 255)
        self.confidence_threshold = 0.7
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ 카메라를 찾을 수 없어요!")
            sys.exit(1)
        print("📹 카메라 준비 완료! ('q' 또는 ESC로 종료)")

    def detect_and_draw(self, frame, results):
        all_normal = True
        for *box, conf, cls in results.xyxy[0].numpy():
            if conf > 0.3:
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                cable_name = self.class_names.get(class_id, "알수없음")
                
                if conf >= self.confidence_threshold:
                    color = self.normal_color
                    status = "정상"
                else:
                    color = self.abnormal_color
                    status = "비정상"
                    all_normal = False
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{cable_name} {status} ({conf:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        msg = "✅ 모든 케이블 정상" if all_normal else "⚠️ 비정상 케이블 감지!"
        msg_color = self.normal_color if all_normal else self.abnormal_color
        cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, msg_color, 3)
        return frame

    def run(self):
        print("🚀 실시간 케이블 체크 시작!")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 카메라 읽기 실패!")
                break
            
            results = self.model(frame)
            frame = self.detect_and_draw(frame, results)
            cv2.imshow('케이블 체크 시스템', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 케이블 체크 시스템 종료!")

def main():
    try:
        checker = RealTimeCableChecker()
        checker.run()
    except KeyboardInterrupt:
        print("\n⚡ 강제 종료!")
    except Exception as e:
        print(f"💥 오류 발생: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

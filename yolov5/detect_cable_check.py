#!/usr/bin/env python3
"""
YOLOv5 Cable Check Detection Script
노트북 케이블 단자 체크를 위한 YOLO 탐지 스크립트

Usage:
    python detect_cable_check.py --source path/to/images --weights runs/train/cable_check/weights/best.pt
"""

import argparse
import os
import sys
import cv2
import json
from pathlib import Path
from datetime import datetime

# Add YOLOv5 root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from detect import main as detect_main, parse_opt as detect_parse_opt


def parse_opt():
    """Parse command line arguments for cable check detection."""
    parser = argparse.ArgumentParser(description='YOLOv5 Cable Check Detection')
    
    # Essential arguments
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/cable_check/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/laptop_cable_check.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='cable_check', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # Cable check specific arguments
    parser.add_argument('--cable-report', action='store_true', help='generate cable connection report')
    parser.add_argument('--alert-threshold', type=float, default=0.7, help='confidence threshold for alerts')
    parser.add_argument('--save-report', action='store_true', help='save detection report as JSON')
    
    return parser.parse_args()


class CableCheckAnalyzer:
    """케이블 연결 상태 분석 클래스"""
    
    def __init__(self):
        self.cable_types = {
            'power': ['power_cable_connected', 'power_cable_disconnected'],
            'usb': ['usb_cable_connected', 'usb_cable_disconnected'],
            'hdmi': ['hdmi_cable_connected', 'hdmi_cable_disconnected'],
            'ethernet': ['ethernet_cable_connected', 'ethernet_cable_disconnected'],
            'audio': ['audio_cable_connected', 'audio_cable_disconnected']
        }
        
        self.class_names = {
            0: 'power_cable_connected',
            1: 'power_cable_disconnected',
            2: 'usb_cable_connected',
            3: 'usb_cable_disconnected',
            4: 'hdmi_cable_connected',
            5: 'hdmi_cable_disconnected',
            6: 'ethernet_cable_connected',
            7: 'ethernet_cable_disconnected',
            8: 'audio_cable_connected',
            9: 'audio_cable_disconnected'
        }
        
        self.reports = []
    
    def analyze_detections(self, detections, image_path):
        """탐지 결과 분석"""
        report = {
            'image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'cable_status': {},
            'alerts': [],
            'summary': {}
        }
        
        # Initialize cable status
        for cable_type in self.cable_types.keys():
            report['cable_status'][cable_type] = 'unknown'
        
        # Process detections
        for detection in detections:
            class_id = int(detection[5])
            confidence = float(detection[4])
            class_name = self.class_names.get(class_id, 'unknown')
            
            # Determine cable type and connection status
            for cable_type, class_list in self.cable_types.items():
                if class_name in class_list:
                    if 'connected' in class_name:
                        report['cable_status'][cable_type] = 'connected'
                    elif 'disconnected' in class_name:
                        report['cable_status'][cable_type] = 'disconnected'
                        # Add alert for disconnected cables
                        report['alerts'].append({
                            'type': 'disconnected_cable',
                            'cable_type': cable_type,
                            'confidence': confidence,
                            'message': f'{cable_type.upper()} cable is disconnected!'
                        })
        
        # Generate summary
        connected_count = sum(1 for status in report['cable_status'].values() if status == 'connected')
        disconnected_count = sum(1 for status in report['cable_status'].values() if status == 'disconnected')
        unknown_count = sum(1 for status in report['cable_status'].values() if status == 'unknown')
        
        report['summary'] = {
            'total_cables': len(self.cable_types),
            'connected': connected_count,
            'disconnected': disconnected_count,
            'unknown': unknown_count,
            'overall_status': 'OK' if disconnected_count == 0 else 'NEEDS_ATTENTION'
        }
        
        self.reports.append(report)
        return report
    
    def print_report(self, report):
        """리포트 출력"""
        print("\n" + "="*50)
        print("🔌 CABLE CONNECTION REPORT")
        print("="*50)
        print(f"📸 Image: {report['image']}")
        print(f"🕐 Time: {report['timestamp']}")
        print("\n📊 Cable Status:")
        
        for cable_type, status in report['cable_status'].items():
            status_icon = "✅" if status == 'connected' else "❌" if status == 'disconnected' else "❓"
            print(f"   {status_icon} {cable_type.upper()}: {status.upper()}")
        
        if report['alerts']:
            print("\n⚠️  ALERTS:")
            for alert in report['alerts']:
                print(f"   - {alert['message']} (Confidence: {alert['confidence']:.2f})")
        
        print(f"\n📈 Summary: {report['summary']['overall_status']}")
        print(f"   Connected: {report['summary']['connected']}/{report['summary']['total_cables']}")
        print("="*50)
    
    def save_reports(self, output_path):
        """리포트를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.reports, f, indent=2, ensure_ascii=False)
        print(f"💾 Reports saved to: {output_path}")


def main(opt):
    """Main detection function for cable check."""
    print("🔍 Starting YOLOv5 Cable Check Detection")
    print("=" * 50)
    
    # Initialize cable analyzer
    analyzer = CableCheckAnalyzer()
    
    # Update opt with cable check specific settings
    opt.data = str(ROOT / 'data' / 'laptop_cable_check.yaml')
    opt.name = 'cable_check_detection'
    
    print(f"🔧 Detection Configuration:")
    print(f"   - Model: {opt.weights}")
    print(f"   - Source: {opt.source}")
    print(f"   - Confidence Threshold: {opt.conf_thres}")
    print(f"   - Alert Threshold: {opt.alert_threshold}")
    print("=" * 50)
    
    # Run detection using YOLOv5's main detection function
    # Note: You would need to modify this to capture detection results
    # For now, we'll use the standard detect function
    detect_main(opt)
    
    print("🎉 Cable Check Detection Completed!")
    
    # If cable report is requested, save it
    if opt.save_report:
        report_path = Path(opt.project) / opt.name / 'cable_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        analyzer.save_reports(report_path)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

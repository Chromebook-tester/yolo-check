# LabelImg 사용 가이드 📖

LabelImg는 YOLO 형식 라벨링을 위한 가장 인기있는 도구야!

## 🚀 실행하기:

### 방법 1: 커맨드라인
```bash
labelImg
```

### 방법 2: 폴더 지정해서 실행
```bash
labelImg [이미지 폴더] [라벨 저장 폴더]
```

## 🎯 사용법:

### 1. 폴더 설정
- **Open Dir**: 이미지가 있는 폴더 선택
- **Change Save Dir**: 라벨 파일 저장할 폴더 선택

### 2. YOLO 형식 설정 ⚠️ 중요!
- 왼쪽 메뉴에서 **"PascalVOC"** 클릭
- **"YOLO"**로 변경 (우리가 쓸 형식)

### 3. 라벨링하기
1. **Create RectBox** (W키) 클릭
2. 마우스로 케이블 주위에 박스 그리기
3. 클래스 선택 (전원연결, USB연결 등)
4. **Save** (Ctrl+S) 저장

### 4. 단축키들:
- **W**: 박스 그리기 모드
- **A**: 이전 이미지
- **D**: 다음 이미지  
- **Del**: 선택된 박스 삭제
- **Ctrl+S**: 저장
- **Ctrl+D**: 현재 박스 복사

## 🔧 클래스 설정:

### classes.txt 파일 만들기:
```
power_cable_connected
power_cable_disconnected
usb_cable_connected
usb_cable_disconnected
hdmi_cable_connected
hdmi_cable_disconnected
ethernet_cable_connected
ethernet_cable_disconnected
audio_cable_connected
audio_cable_disconnected
```

이 파일을 이미지 폴더에 넣으면 자동으로 클래스 목록이 나와!

## 🎉 장점들:

### ✅ LabelImg 장점:
- **직관적인 인터페이스**
- **YOLO 형식 바로 저장**
- **단축키로 빠른 작업**
- **박스 크기 조정 쉬움**
- **많은 사람들이 사용**

### 🤔 내가 만든 것과 비교:
- **LabelImg**: 정교한 박스 그리기, 전문적
- **내 프로그램**: 간단하고 빠름, 학습용

## 🎯 추천 작업 흐름:

### 정교한 라벨링이 필요하면:
```bash
labelImg raw_images raw_labels
```

### 빠른 테스트용이면:
```bash
python simple_labeler.py
```

## 💡 팁:

1. **박스 크기**: 케이블보다 조금 크게 그리기
2. **일관성**: 같은 케이블은 항상 같은 크기로
3. **정확성**: 케이블이 화면에 완전히 보일 때만 라벨링
4. **저장**: 자주자주 저장하기 (Ctrl+S)

---

**두 도구 다 써보고 편한 걸로 선택해! 😊**

@echo off
echo 🔌 실시간 케이블 체크 시작!
echo.
echo 사용법:
echo 1. 웹캠이 연결되어 있는지 확인
echo 2. 노트북을 카메라에 비추기
echo 3. 초록박스 = 정상연결, 빨간박스 = 문제있음
echo 4. q 키를 눌러서 종료
echo.
pause

python realtime_cable_check.py

pause

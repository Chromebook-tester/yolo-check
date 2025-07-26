@echo off
python train_cable_check.py --epochs 100 --batch-size 8 --data data/laptop_cable_check.yaml --weights yolov5s.pt --name cable_check
pause
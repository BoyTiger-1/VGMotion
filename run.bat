@echo off
cd /d %~dp0
set TF_ENABLE_ONEDNN_OPTS=0
python -m motionforge %*

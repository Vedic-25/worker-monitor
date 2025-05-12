My script: Loads a video of a worker and Detects human pose in each frame using MediaPipe it also Determines if the worker is “Working” or “Idle” based on posture and motion. Adds some time-based smoothing to prevent flickering labels.Logs any status changes to a CSV file with timestamps.
Determines if the current pose represents "working" based on:
A. Keypoints Used
Wrists, elbows, and shoulders (left and right)

B. Pose Confidence Filter
Skips frame if keypoints are not clearly visible.

C. Posture Detection
Detects active posture via elbow angles or wrist positions.

D. Motion Detection
If wrist moved → motion-based engagement.

E. Return Result

These libraries support:

cv2: Reading and displaying video.
mediapipe: Human pose estimation.
numpy: Vector math.
csv: Logging status changes.
datetime: Timestamps for logs

I used Windows but for automatically starting the script you can use these commands in ubuntu:-

1. For creating file:-
sudo nano /etc/systemd/system/worker_monitor.service

2. Add these steps in that file:-

[Unit]
Description=Factory Worker Activity Monitor
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/ubuntu123/Desktop/factory_worker/main.py
WorkingDirectory=/home/ubuntu123/Desktop/factory_worker
StandardOutput=file:/var/log/worker_monitor.log
StandardError=file:/var/log/worker_monitor.err
Restart=always
User=ubuntu123
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target


3. use this commands to execute:-
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable worker_monitor.service
sudo systemctl start worker_monitor.service


4.Check status:
sudo systemctl status worker_monitor.service

5.View logs live:
sudo tail -f /var/log/worker_monitor.log



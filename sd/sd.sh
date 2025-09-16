import os

# Quay lại /content để tránh lỗi "getcwd"
os.chdir("/content")

# Clone repo nếu chưa có
if not os.path.exists("BayesDiff"):
    !git clone https://github.com/some-repo/BayesDiff.git
else:
    print("Repo BayesDiff đã tồn tại.")

# Tạo venv mới cho Python 3.9
!python3.9 -m venv bayesdiff_env

# Ghi file sd.sh trong /content
sd_sh_content = """#!/bin/bash
set -e

# Quay lại /content để chắc chắn
cd /content

# Kích hoạt env
source bayesdiff_env/bin/activate

# Cài đặt phụ thuộc
pip install --upgrade pip setuptools wheel
pip install numpy==1.23.5 pandas==1.5.3 scipy==1.10.1

# Vào thư mục code
cd BayesDiff

# Chạy thử
python setup.py install
"""

with open("/content/sd.sh", "w") as f:
    f.write(sd_sh_content)

# Cấp quyền chạy
!chmod +x /content/sd.sh

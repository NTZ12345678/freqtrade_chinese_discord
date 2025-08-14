import subprocess

# 定义命令和日志文件路径
command = [
    "freqtrade",
    "backtesting",
    "--config", "user_data/config/spot_ma.json",
    "--timerange", "20210102-20250801",
    "--breakdown", "month",
]



log_file = "user_data/logs/ma/test25.log"



# 打开日志文件并运行命令
with open(log_file, "w", encoding="utf-8") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

print(f"Process started with PID: {process.pid}")
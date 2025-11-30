import subprocess
import os

base = r"C:\ai_scalper_bot"
py = os.path.join(base, "venv", "Scripts", "python.exe")
cmd = [py, "-m", "bot.market_data.ws_manager"]

subprocess.call(cmd)

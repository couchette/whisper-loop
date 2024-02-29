from ctypes import wintypes, windll
import os

if __name__ == "__main__":
    os.system(
        "powershell -Command \"Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))\""
    )

    os.system("choco install ffmpeg -y")

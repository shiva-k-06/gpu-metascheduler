import subprocess
import json


def detect_vendor(name):
    name_lower = name.lower()

    if "nvidia" in name_lower:
        return "NVIDIA"
    if "intel" in name_lower:
        return "Intel"
    if "amd" in name_lower or "radeon" in name_lower:
        return "AMD"

    return "Unknown"


def get_windows_gpus():
    command = [
        "powershell",
        "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor | "
        "ConvertTo-Json"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    data = json.loads(result.stdout)

    if isinstance(data, dict):
        data = [data]

    gpus = []

    for item in data:
        name = item.get("Name", "Unknown GPU")
        memory_bytes = item.get("AdapterRAM", 0) or 0
        memory_gb = round(memory_bytes / (1024 ** 3), 2)

        vendor = detect_vendor(name)

        gpus.append({
            "name": name,
            "vendor": vendor,
            "memory_gb": memory_gb,
            "driver_version": item.get("DriverVersion"),
            "video_processor": item.get("VideoProcessor"),
            "speed_score": estimate_speed_score(name, memory_gb)
        })

    return gpus


def estimate_speed_score(name, memory_gb):
    name_lower = name.lower()

    if "rtx" in name_lower:
        return 1.0
    if "gtx" in name_lower:
        return 0.75
    if "radeon" in name_lower or "rx" in name_lower:
        return 0.75
    if "iris" in name_lower:
        return 0.45
    if "uhd" in name_lower or "intel" in name_lower:
        return 0.35

    if memory_gb >= 8:
        return 0.8
    if memory_gb >= 4:
        return 0.6

    return 0.4


if __name__ == "__main__":
    gpus = get_windows_gpus()

    print("\n=== Detected GPUs ===\n")

    for i, gpu in enumerate(gpus):
        print(f"GPU {i}")
        print("Name           :", gpu["name"])
        print("Vendor         :", gpu["vendor"])
        print("Memory (GB)    :", gpu["memory_gb"])
        print("Driver Version :", gpu["driver_version"])
        print("Processor      :", gpu["video_processor"])
        print("Speed Score    :", gpu["speed_score"])
        print()
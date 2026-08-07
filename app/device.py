from __future__ import annotations

import re
import subprocess
from io import BytesIO

from PIL import Image


class AdbController:
    def __init__(self, adb_path: str, device_serial: str | None = None) -> None:
        self.adb_path = adb_path
        self.device_serial = device_serial
        self._ensure_connected()

    def screenshot(self) -> Image.Image | None:
        command = self._command(["exec-out", "screencap", "-p"])
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0:
            print(completed.stderr.decode(errors="replace").strip())
            return None
        return Image.open(BytesIO(completed.stdout)).convert("RGBA")

    def click(self, x: int, y: int) -> None:
        subprocess.run(self._command(["shell", "input", "tap", str(x), str(y)]), check=False)

    def _ensure_connected(self) -> None:
        completed = subprocess.run(self._command(["devices"]), capture_output=True, text=True, check=False)
        devices = [line for line in completed.stdout.splitlines()[1:] if re.search(r"\tdevice$", line)]
        if not devices:
            raise RuntimeError("No Android device is connected through ADB")

    def _command(self, arguments: list[str]) -> list[str]:
        command = [self.adb_path]
        if self.device_serial:
            command.extend(["-s", self.device_serial])
        return command + arguments

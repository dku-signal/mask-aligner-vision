"""Count acknowledged motor steps relative to the taught upper-right position."""
import json
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile

DIRECTIONS = {"F": ("X", 1), "B": ("X", -1), "f": ("Y", 1), "b": ("Y", -1)}
STATE = Path(__file__).resolve().parents[1] / "out" / "fine_stage.json"


class FineStage:
    def __init__(self, device, path=STATE):
        self.device, self.path = device, path
        self.lock = threading.RLock()
        self.data = {}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("device") == device:
                self.data = data

    @classmethod
    def for_port(cls, port):
        from serial.tools import list_ports
        if port == "mock":
            return cls(port, STATE.with_name("fine_stage_mock.json"))
        device = next((p.serial_number for p in list_ports.comports() if p.device == port), None)
        return cls(device or port)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", dir=self.path.parent, delete=False, encoding="utf-8") as f:
            json.dump(self.data, f)
        Path(f.name).replace(self.path)

    def teach(self):
        with self.lock:
            self.data = dict(device=self.device, offset={"X": 0, "Y": 0}, pending=None, known=True)
            self.save()

    def invalidate(self):
        with self.lock:
            if self.data.get("known"):
                self.data["known"] = False
                self.save()

    def begin(self, letter, steps):
        with self.lock:
            if not self.data.get("known"):
                return
            if self.data["pending"] is not None:
                self.invalidate()
                return
            self.data["pending"] = [letter, steps]
            self.save()

    def ack(self, letter, steps, partial=False):
        with self.lock:
            if not self.data.get("known"):
                return
            pending = self.data["pending"]
            if not pending or pending[0] != letter or not (steps == pending[1] or partial and 0 <= steps <= pending[1]):
                self.invalidate()
                return
            axis, sign = DIRECTIONS[letter]
            self.data["offset"][axis] += sign * steps
            self.data["pending"] = None
            self.save()

    def position(self):
        with self.lock:
            if not self.data.get("known") or self.data["pending"] is not None:
                raise ValueError("정밀 X·Y를 오른쪽 위에 맞춘 뒤 현재 위치를 기준으로 저장하세요")
            return self.data["offset"]["X"], self.data["offset"]["Y"]

    def moves(self, target_x=0, target_y=0):
        current_x, current_y = self.position()
        moves = []
        for delta, letters in ((target_x - current_x, "FB"), (target_y - current_y, "fb")):
            if delta:
                moves.append((letters[delta < 0], abs(delta)))
        return moves

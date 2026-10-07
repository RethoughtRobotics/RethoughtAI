from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parents[3] / "assets"
TOPIC  = "/robot/xdisplay"
FONTS  = Path("/usr/share/fonts/truetype/liberation")


class BaxterScreen:
    """Image screens from assets/ (trigger.screen: true)."""

    def __init__(self, node, session=None):
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from sensor_msgs.msg import Image
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = node.create_publisher(Image, TOPIC, qos)
        self.episode = 0

    def idle(self):        self._show("home.jpg")
    def recording(self):   self._show("recording.jpg")
    def decide(self, length=None, frames=None): self._show("save_or_discard.jpg")
    def saving(self):  self._show("saving.jpg")
    def discarding(self):  self._show("discarding.jpg")

    def _show(self, filename: str) -> None:
        path = ASSETS / filename
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Baxter screen asset not found: {path}")
        cv2.putText(img, f"Ep {self.episode:03d}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        self._publish(img)

    def _publish(self, img) -> None:
        from sensor_msgs.msg import Image
        msg = Image()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = img.shape[1] * img.shape[2]
        msg.data = img.tobytes()
        self._pub.publish(msg)


class SessionScreen(BaxterScreen):
    """Drawn screens (trigger.screen: session): the session name, the episode, and the buttons
    for this step; the one the user picked turns green."""

    SIZE   = (1024, 600)  # Baxter's display
    GREEN  = (46, 158, 68)
    GREY   = (100, 100, 100)
    LINE   = (70, 70, 70)

    def __init__(self, node, session=None):
        super().__init__(node)
        self.session = session or "Session"
        self._info = []

    def idle(self):       self._draw(("Start", "Stop"))
    def recording(self):  self._draw(("Start", "Stop"), selected="Start")
    def decide(self, length=None, frames=None):
        info = [f"Length: {length:.2f} seconds" if length is not None else None,
                f"Frames: {frames}" if frames is not None else None]
        self._info = [i for i in info if i]  # kept on the saving/discarding screen
        self._draw(("DISCARD", "SAVE"), info=self._info)
    def saving(self):     self._draw(("DISCARD", "SAVE"), selected="SAVE", info=self._info)
    def discarding(self): self._draw(("DISCARD", "SAVE"), selected="DISCARD", info=self._info)

    def _draw(self, buttons, selected=None, info=()):
        from PIL import Image, ImageDraw, ImageFont

        def font(bold, size):
            try:
                return ImageFont.truetype(str(FONTS / f"LiberationSans-{'Bold' if bold else 'Regular'}.ttf"), size)
            except OSError:
                return ImageFont.load_default(size)

        img = Image.new("RGB", self.SIZE, "white")
        d = ImageDraw.Draw(img)
        d.text((60, 50), self.session, fill="black", font=font(True, 96))
        d.text((60, 160), f"Episode {self.episode:03d}", fill=self.GREY, font=font(True, 64))
        for i, line in enumerate(info):
            d.text((64, 270 + 60 * i), line, fill="black", font=font(True, 40))
        label = font(False, 44)
        for i, name in enumerate(buttons):
            box = (60 + 340 * i, 430, 360 + 340 * i, 530)
            on = name == selected
            d.rectangle(box, fill=self.GREEN if on else "white", outline=self.GREEN if on else self.LINE, width=3)
            d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), name,
                   fill="white" if on else "black", font=label, anchor="mm")
        self._publish(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))


def render_previews(out_dir=ASSETS / "session", session="CupInCube"):
    """Write each SessionScreen state as a PNG (example data), without ROS:
    python -m recorder.screen [out_dir]"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    screen = SessionScreen.__new__(SessionScreen)
    screen.session, screen.episode, screen._info = session, 1, []
    for name, show in [("idle", screen.idle), ("recording", screen.recording),
                       ("save_or_discard", lambda: screen.decide(45.23, 1200)),
                       ("saving", screen.saving), ("discarding", screen.discarding)]:
        screen._publish = lambda img, name=name: cv2.imwrite(str(out_dir / f"{name}.png"), img)
        show()


if __name__ == "__main__":
    import sys
    render_previews(*sys.argv[1:2])

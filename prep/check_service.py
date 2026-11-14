"""Try GetClassifications with a preprocessed image; fall back notes for ov-anomaly."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prep.preprocess import preprocess


async def main():
    from viam.robot.client import RobotClient
    from viam.services.vision import VisionClient
    from viam.media.video import CameraMimeType, ViamImage

    # pick a sample
    samples = list((ROOT / "train_images" / "good").glob("*.jpg"))
    if not samples:
        print("no train_images/good samples")
        return
    bgr = cv2.imread(str(samples[0]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out, dbg = preprocess(rgb)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    assert ok
    image = ViamImage(buf.tobytes(), CameraMimeType.PNG)

    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    machine = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    try:
        anomaly = VisionClient.from_robot(machine, "anomaly")
        cls = await anomaly.get_classifications(image, 1)
        print("get_classifications OK:", [(c.class_name, c.confidence) for c in cls])
        print("preprocess dbg:", dbg)
    except Exception as e:
        print("get_classifications FAILED:", repr(e))
        print("Fall back: use ov-anomaly mlmodel.infer() with preprocessed tensor.")
        try:
            from viam.services.mlmodel import MLModelClient
            ml = MLModelClient.from_robot(machine, "ov-anomaly")
            meta = await ml.metadata()
            print("ov-anomaly metadata:", meta)
        except Exception as e2:
            print("ov-anomaly metadata also failed:", repr(e2))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())

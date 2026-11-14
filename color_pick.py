"""
Colour-targeted pick for armfarm14 (Viam "hackathons" org, Arm Farm).

Runs as-is in the Control tab's code sandbox, which injects
VIAM_API_KEY, VIAM_API_KEY_ID and VIAM_MACHINE_ADDRESS.

To run locally instead:
  pip install viam-sdk
  export VIAM_API_KEY=...            # CONNECT tab > API keys
  export VIAM_API_KEY_ID=...
  export VIAM_MACHINE_ADDRESS=armfarm14-main.310sld03v2.viam.cloud
  python color_pick.py

Set COLOR below, run once with EXECUTE = False to check the printed plan,
then set EXECUTE = True to move the arm.
"""
import asyncio
import os

from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

# Run settings
COLOR = "yellow"          # red, orange, yellow, green, blue, purple
DETECT_FROM_HOME = True   # False = detect from wherever the arm is now
EXECUTE = False           # False = detect and print the plan only, no motion

# Tunables
APPROACH_MM = -100        # standoff 100 mm toward the camera from the object centre
GRIPPER_LENGTH_MM = -60   # fingertips sit 60 mm past the gripper TCP
EXTRA_DEPTH_MM = 0        # raise if the fingers close above the object
MAX_GRIP_MM = 80          # UFactory gripper opens to about 85 mm
JAW_OFFSET_DEG = -90      # theta that keeps the wrist where it is now; tune once
ALIGN_TO_OBJECT = True    # rotate the wrist to the object's on-screen angle

CAM = "cam"
GRIPPER = "gripper"
COLORS = {"red", "orange", "yellow", "green", "blue", "purple"}


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)


def to_pixel(p, intr):
    """Project a cam-frame point (mm) onto the colour image."""
    if p.z <= 0:
        return None
    u = intr.focal_x_px * p.x / p.z + intr.center_x_px
    v = intr.focal_y_px * p.y / p.z + intr.center_y_px
    return u, v


async def object_angle(geom, detector, intr):
    """Find the 2D detection behind this segment and return its rotation."""
    px = to_pixel(geom.center, intr)
    if px is None:
        return 0.0
    u, v = px
    result = await detector.do_command({"command": "detect", "camera": CAM})
    for d in result.get("detections", []):
        inside = d["x_min"] <= u <= d["x_max"] and d["y_min"] <= v <= d["y_max"]
        if d["label"] == geom.label and inside:
            return float(d["angle_deg"])
    return 0.0


async def find_target(color, vision, detector, intr):
    objects = await vision.get_object_point_clouds(CAM)
    labels = [o.geometries.geometries[0].label for o in objects]
    matches = [o for o in objects
               if o.geometries.geometries[0].label.endswith("-" + color)]
    if not matches:
        print(f"No {color} object in view. Seen: {labels or 'nothing'}")
        return None

    # Largest point cloud = closest or most fully visible object
    obj = max(matches, key=lambda o: len(o.point_cloud))
    geom = obj.geometries.geometries[0]

    if geom.WhichOneof("geometry_type") == "box":
        d = geom.box.dims_mm
        width = min(d.x, d.y)
        print(f"{geom.label}: {d.x:.0f} x {d.y:.0f} x {d.z:.0f} mm")
        if width > MAX_GRIP_MM:
            print(f"Too wide for the gripper ({width:.0f} mm > {MAX_GRIP_MM} mm)")
            return None

    angle = await object_angle(geom, detector, intr) if ALIGN_TO_OBJECT else 0.0
    return geom, angle


def gripper_z(mm):
    """Straight move along the gripper's own axis, no rotation."""
    return PoseInFrame(
        reference_frame=GRIPPER,
        pose=Pose(x=0, y=0, z=mm, o_x=0, o_y=0, o_z=1, theta=0),
    )


async def main():
    if COLOR not in COLORS:
        raise ValueError(f"COLOR must be one of {sorted(COLORS)}")

    async with await connect() as machine:
        names = {r.name for r in machine.resource_names}
        gripper = Gripper.from_robot(machine, GRIPPER)
        home = Switch.from_robot(machine, "home-pose")
        cam = Camera.from_robot(machine, CAM)
        vision = VisionClient.from_robot(machine, "vision-segment")
        detector = VisionClient.from_robot(machine, "shape-detector")
        motion = MotionClient.from_robot(machine, "builtin")
        intr = (await cam.get_properties()).intrinsic_parameters

        if DETECT_FROM_HOME:
            if not EXECUTE:
                print("EXECUTE is False: detecting from the current pose, arm not moved.")
            else:
                await home.set_position(2)
                await asyncio.sleep(0.5)  # settle before detecting

        found = await find_target(COLOR, vision, detector, intr)
        if found is None:
            return
        geom, angle = found

        c = geom.center
        approach = Pose(x=c.x, y=c.y, z=c.z + APPROACH_MM,
                        o_x=0, o_y=0, o_z=1, theta=JAW_OFFSET_DEG + angle)
        descend_mm = (APPROACH_MM - GRIPPER_LENGTH_MM) * -1 + EXTRA_DEPTH_MM

        print(f"Target {geom.label} at cam ({c.x:.0f}, {c.y:.0f}, {c.z:.0f}) mm, "
              f"angle {angle:.1f} deg")
        print(f"Approach z {approach.z:.0f} mm, theta {approach.theta:.1f}, "
              f"then descend {descend_mm:.0f} mm")
        if not EXECUTE:
            print("Dry run done. Set EXECUTE = True to pick.")
            return

        # Approach in the cam frame while the camera is still where it detected
        await motion.move(component_name=GRIPPER,
                          destination=PoseInFrame(reference_frame=CAM, pose=approach))
        await gripper.open()
        await asyncio.sleep(0.3)

        # Descend in the gripper frame so camera motion no longer matters
        await motion.move(component_name=GRIPPER, destination=gripper_z(descend_mm))
        grabbed = await gripper.grab()
        await asyncio.sleep(0.3)
        print("Grabbed" if grabbed else "Gripper closed on nothing")

        await motion.move(component_name=GRIPPER, destination=gripper_z(-descend_mm))

        if "place-pose" in names:
            await Switch.from_robot(machine, "place-pose").set_position(2)
            await gripper.open()
            await home.set_position(2)
        else:
            await home.set_position(2)
            print("No place-pose configured; holding the object at home.")


if __name__ == "__main__":
    asyncio.run(main())

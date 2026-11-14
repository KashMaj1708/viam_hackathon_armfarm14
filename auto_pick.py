"""
Automatic pick for armfarm14 (Viam "hackathons" org, Arm Farm).

Finds whatever object is on the table with 3D clustering (no colour or
shape model needed), then picks it top-down. Works for the black ball,
blocks, anything that fits the gripper.

Pipeline (all configured on the machine):
  cam            RealSense on the wrist
  cam-workspace  erh:vmodutils:pc-crop-camera, crops the depth cloud to the
                 table area in WORLD coordinates, starting 15 mm above the table
  object-cluster erh:vmodutils:pc-cluster, groups the cropped points into objects

Because cam-workspace crops in the world frame, every object centre comes
back in world millimetres, so the grasp is planned straight in "world".

Run from the Viam app Control tab code sandbox (injects the three env vars)
or locally:
  pip install viam-sdk
  set VIAM_API_KEY, VIAM_API_KEY_ID, VIAM_MACHINE_ADDRESS
  python auto_pick.py
python auto_pick.py            -> dry run, prints the plan, arm does not move
python auto_pick.py --execute  -> picks
"""
import asyncio
import os
import sys

from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

EXECUTE = False   # set True to actually pick (or run locally with --execute)
EXECUTE = EXECUTE or "--execute" in sys.argv

# Geometry, measured on this station (Nov 13)
TABLE_Z = -10           # real table top in world z (mm)
TIP_MM = 60             # fingertips sit this far past the gripper frame origin
APPROACH_MM = 100       # standoff above the grasp height
LIFT_Z = 320            # gripper height after grabbing (clears the box walls)
GRASP_FRACTION = 0.45   # close the fingers at this fraction of object height
MIN_TIP_CLEARANCE = 8   # never put fingertips closer than this to the table
MAX_GRIP_MM = 80        # gripper opens to about 84 mm
MIN_HEIGHT_MM = 12      # ignore flat things (paper, tape, cables)
DROP_XY = (300, 230)    # cardboard box on the left of the table; None = hold at home
DROP_HIGH_Z = 320       # travel height over the box (walls are ~100 mm tall)
DROP_Z = 250            # release height, ball clears the rim and falls in

GRIPPER = "gripper"
DOWN = dict(o_x=0, o_y=0, o_z=-1)   # gripper pointing straight down
THETA_JAWS_Y = -180     # jaws close along world y (verified pose)
THETA_JAWS_X = -90      # rotated 90 deg: jaws close along world x


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)


def world(x, y, z, theta):
    return PoseInFrame(reference_frame="world",
                       pose=Pose(x=x, y=y, z=z, theta=theta, **DOWN))


def describe(obj):
    g = obj.geometries.geometries[0]
    c, d = g.center, g.box.dims_mm
    top = c.z + d.z / 2
    return {
        "x": c.x, "y": c.y, "top": top,
        "height": top - TABLE_Z,
        "dx": d.x, "dy": d.y,
        "width": min(d.x, d.y),
        "points": len(obj.point_cloud),
    }


def choose(objects):
    cands = []
    for o in objects:
        info = describe(o)
        why = None
        if info["height"] < MIN_HEIGHT_MM:
            why = "too flat"
        elif info["width"] > MAX_GRIP_MM:
            why = "too wide"
        print(f"  object at ({info['x']:.0f}, {info['y']:.0f}) "
              f"{info['dx']:.0f}x{info['dy']:.0f} mm, top z {info['top']:.0f}, "
              f"{info['points']} pts{'  -> skip, ' + why if why else ''}")
        if not why:
            cands.append(info)
    # biggest graspable object = most fully seen, least likely to be noise
    return max(cands, key=lambda i: i["points"]) if cands else None


async def main():
    async with await connect() as machine:
        gripper = Gripper.from_robot(machine, GRIPPER)
        home = Switch.from_robot(machine, "home-pose")
        finder = VisionClient.from_robot(machine, "object-cluster")
        motion = MotionClient.from_robot(machine, "builtin")

        if EXECUTE:
            await gripper.open()
            await home.set_position(2)
            await asyncio.sleep(1.0)      # let the arm settle before looking
        else:
            print("Dry run: detecting from the current pose, arm not moved.")

        objects = await finder.get_object_point_clouds("cam-workspace")
        print(f"Found {len(objects)} object(s):")
        t = choose(objects)
        if t is None:
            print("Nothing graspable in view.")
            return

        tip_z = max(TABLE_Z + t["height"] * GRASP_FRACTION,
                    TABLE_Z + MIN_TIP_CLEARANCE)
        grasp_z = tip_z + TIP_MM
        # jaws should close across the object's narrower side
        theta = THETA_JAWS_X if t["dx"] < t["dy"] * 0.8 else THETA_JAWS_Y

        print(f"Target ({t['x']:.0f}, {t['y']:.0f}), width {t['width']:.0f} mm, "
              f"height {t['height']:.0f} mm")
        print(f"Approach z {grasp_z + APPROACH_MM:.0f}, grasp z {grasp_z:.0f} "
              f"(fingertips at {tip_z:.0f}), theta {theta}")
        if not EXECUTE:
            print("Dry run only. Rerun with --execute to pick.")
            return

        await motion.move(GRIPPER, world(t["x"], t["y"], grasp_z + APPROACH_MM, theta))
        await motion.move(GRIPPER, world(t["x"], t["y"], grasp_z, theta))
        await gripper.grab()
        await asyncio.sleep(0.3)
        await motion.move(GRIPPER, world(t["x"], t["y"], LIFT_Z, theta))

        if not await gripper.is_holding_something():
            print("Missed: gripper closed on nothing. Opening and going home.")
            await gripper.open()
            await home.set_position(2)
            return
        print("Holding the object.")

        if DROP_XY:
            print(f"Dropping into the box at {DROP_XY}.")
            await motion.move(GRIPPER, world(DROP_XY[0], DROP_XY[1], DROP_HIGH_Z, theta))
            await motion.move(GRIPPER, world(DROP_XY[0], DROP_XY[1], DROP_Z, theta))
            await gripper.open()
            await asyncio.sleep(0.5)
            await motion.move(GRIPPER, world(DROP_XY[0], DROP_XY[1], DROP_HIGH_Z, theta))
        await home.set_position(2)


if __name__ == "__main__":
    asyncio.run(main())

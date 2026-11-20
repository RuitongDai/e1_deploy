"""Convert the training URDF to a portable MuJoCo scene with mixed servos."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e1_wheel.wheel_config import (
    BASE_HEIGHT,
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    PHYSICS_DT,
    SCENE_PATH,
    URDF_PATH,
)


def build_scene(urdf_path: Path = URDF_PATH, output_path: Path = SCENE_PATH) -> Path:
    urdf_path, output_path = Path(urdf_path).resolve(), Path(output_path).resolve()
    source = ET.parse(urdf_path).getroot()
    compiler = source.find("mujoco/compiler")
    if compiler is None:
        extension = source.find("mujoco")
        if extension is None:
            extension = ET.SubElement(source, "mujoco")
        compiler = ET.SubElement(extension, "compiler")
    compiler.set("meshdir", str(urdf_path.parent / "meshes"))
    compiler.set("strippath", "true")
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))
    with tempfile.TemporaryDirectory(prefix="e1-wheel-mjcf-") as temp_dir:
        intermediate = Path(temp_dir) / "imported.xml"
        mujoco.mj_saveLastXML(str(intermediate), imported)
        scene = ET.parse(intermediate).getroot()

    # Save paths relative to the scene, so the whole e1_wheel directory is portable.
    scene.find("compiler").set(
        "meshdir", os.path.relpath(urdf_path.parent / "meshes", output_path.parent)
    )
    for mesh in scene.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    option = scene.find("option")
    if option is None:
        option = ET.SubElement(scene, "option")
    option.attrib.update(
        timestep=str(PHYSICS_DT),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        solver="Newton",
        iterations="100",
        ls_iterations="50",
        tolerance="1e-6",
        cone="pyramidal",
    )
    world = scene.find("worldbody")
    base = world.find("body[@name='base_link']")
    if base is None:
        raise ValueError("URDF conversion did not preserve base_link")
    base.set("pos", f"0 0 {BASE_HEIGHT}")
    ET.SubElement(base, "freejoint", name="floating_base")

    # Training disables self-collisions. Only robot <-> floor pairs are enabled.
    # Keep precisely the authored URDF collisions; visual-only links stay visual.
    for geom in world.iter("geom"):
        visual = geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0"
        geom.set("group", "1" if visual else "3")
        geom.set("contype", "0" if visual else "1")
        geom.set("conaffinity", "0")
        if not visual:
            geom.set("friction", "0.8 0.005 0.0001")
            geom.set("condim", "3")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        contype="0",
        conaffinity="1",
        friction="0.8 0.005 0.0001",
        material="ground",
        rgba="0.35 0.4 0.45 1",
    )
    ET.SubElement(world, "light", pos="0 -3 4", dir="0 0 -1", directional="true")
    asset = scene.find("asset")
    if asset is None:
        asset = ET.SubElement(scene, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="checker",
        type="2d",
        builtin="checker",
        rgb1="0.2 0.25 0.3",
        rgb2="0.3 0.35 0.4",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="ground",
        texture="checker",
        texrepeat="5 5",
        texuniform="true",
        reflectance="0.1",
    )
    actuators = ET.SubElement(scene, "actuator")
    for index, name in enumerate(JOINT_NAMES):
        attributes = dict(
            name=f"servo_{name}",
            joint=name,
            forcelimited="true",
            forcerange=f"{-EFFORT_LIMIT[index]} {EFFORT_LIMIT[index]}",
        )
        if index < 16:
            ET.SubElement(
                actuators,
                "position",
                **attributes,
                kp=str(KP[index]),
                kv=str(KD[index]),
            )
        else:
            ET.SubElement(actuators, "velocity", **attributes, kv=str(KD[index]))
    ET.indent(scene)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(scene).write(output_path, encoding="unicode", xml_declaration=True)
    # Compile once here so a broken generated scene is caught before deployment.
    checked = mujoco.MjModel.from_xml_path(str(output_path))
    if checked.nq != 27 or checked.nv != 26 or checked.nu != 20:
        raise ValueError(
            f"Unexpected model dimensions: nq={checked.nq}, nv={checked.nv}, nu={checked.nu}"
        )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=URDF_PATH)
    parser.add_argument("--output", type=Path, default=SCENE_PATH)
    args = parser.parse_args()
    print(f"Scene written: {build_scene(args.urdf, args.output)}")

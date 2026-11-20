"""SDL game-controller input, independent of MuJoCo's window focus."""

from __future__ import annotations

import os
import time

import numpy as np


class GamepadDisconnected(RuntimeError):
    pass


class Gamepad:
    def __init__(self, device_index: int = 0, deadzone: float = 0.15):
        # Must be set before SDL initialization. The visible window belongs to
        # GLFW/MuJoCo, so Pygame must accept input while it is in the background.
        os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        from pygame._sdl2 import controller

        self.pygame, self.controller_module = pygame, controller
        self.pad = None
        self.joystick = None
        self.deadzone = deadzone
        self.previous = set()
        self.raw_axes = []
        self.raw_buttons = []
        self.held_names = []
        try:
            pygame.display.init()
            pygame.joystick.init()
            controller.init()
            controller.set_eventstate(True)
            count = pygame.joystick.get_count()
            if not 0 <= device_index < count:
                raise RuntimeError(
                    f"SDL sees {count} gamepads; cannot open index {device_index}. "
                    "Check the USB connection and access to /dev/input, then run --test-gamepad."
                )
            self.joystick = pygame.joystick.Joystick(device_index)
            self.instance_id = self.joystick.get_instance_id()
            if not controller.is_controller(device_index):
                raise RuntimeError(
                    f"SDL has no standard controller mapping for {self.joystick.get_name()} "
                    f"(GUID {self.joystick.get_guid()}). Supply this device's mapping via "
                    "SDL_GAMECONTROLLERCONFIG before starting the script."
                )
            self.pad = controller.Controller(device_index)
            self.axes = (
                pygame.CONTROLLER_AXIS_LEFTY,
                pygame.CONTROLLER_AXIS_LEFTX,
                pygame.CONTROLLER_AXIS_RIGHTX,
            )
            self.buttons = {
                pygame.CONTROLLER_BUTTON_X: "stand",
                pygame.CONTROLLER_BUTTON_A: "start",
                pygame.CONTROLLER_BUTTON_Y: "damp",
                pygame.CONTROLLER_BUTTON_B: "exit",
                pygame.CONTROLLER_BUTTON_BACK: "reset",
            }
            self.names = {
                pygame.CONTROLLER_BUTTON_X: "X",
                pygame.CONTROLLER_BUTTON_A: "A",
                pygame.CONTROLLER_BUTTON_Y: "Y",
                pygame.CONTROLLER_BUTTON_B: "B",
                pygame.CONTROLLER_BUTTON_BACK: "Back",
            }
            print(
                f"[GAMEPAD] #{device_index}: {self.joystick.get_name()}, GUID={self.joystick.get_guid()}"
            )
            print(f"[GAMEPAD] SDL standard mapping: {self.pad.get_mapping()}")
            print(
                "[GAMEPAD] Background input enabled. X=stand, A=inference, Y=damping, B=exit, Back=reset"
            )
        except Exception:
            self.close()
            raise

    def read(self):
        pg = self.pygame
        pressed = set()
        try:
            for event in pg.event.get():
                # JOY and CONTROLLER events can describe the same device; only
                # consume standard controller button events to avoid duplicates.
                if getattr(event, "instance_id", None) != self.instance_id:
                    continue
                if event.type in (pg.JOYDEVICEREMOVED, pg.CONTROLLERDEVICEREMOVED):
                    raise GamepadDisconnected("Gamepad disconnected")
                if (
                    event.type == pg.CONTROLLERBUTTONDOWN
                    and event.button in self.buttons
                ):
                    pressed.add(self.buttons[event.button])
            if not self.pad.attached():
                raise GamepadDisconnected("Gamepad disconnected")
            values = np.array(
                [self.pad.get_axis(axis) for axis in self.axes], dtype=float
            )
            values /= np.where(values < 0, 32768.0, 32767.0)
            values[np.abs(values) < self.deadzone] = 0.0
            command = np.clip(-values, -1.0, 1.0)
            held_buttons = {
                button for button in self.buttons if self.pad.get_button(button)
            }
            held = {self.buttons[button] for button in held_buttons}
            # Polling covers already-held buttons; queued DOWN events preserve
            # short taps that were released between two 50-Hz simulation ticks.
            pressed |= held - self.previous
            self.previous = held
            self.held_names = [
                self.names[button] for button in self.buttons if button in held_buttons
            ]
            self.raw_axes = [
                round(self.joystick.get_axis(i), 3)
                for i in range(self.joystick.get_numaxes())
            ]
            self.raw_buttons = [
                i
                for i in range(self.joystick.get_numbuttons())
                if self.joystick.get_button(i)
            ]
            return command, pressed
        except pg.error as error:
            raise GamepadDisconnected(f"Cannot read gamepad: {error}") from error

    def close(self):
        if self.pad is not None:
            self.pad.quit()
        if self.joystick is not None:
            self.joystick.quit()
        self.controller_module.quit()
        self.pygame.joystick.quit()
        self.pygame.display.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def monitor(gamepad: Gamepad, duration: float = 0.0):
    """Print actual input without loading the policy or opening a simulator."""
    print("[GAMEPAD] Move both sticks and press X/A/Y/Back. B or Ctrl+C exits.")
    start = time.monotonic()
    last_print = -1.0
    try:
        while duration <= 0 or time.monotonic() - start < duration:
            command, events = gamepad.read()
            now = time.monotonic()
            if events or now - last_print >= 0.2:
                print(
                    f"[GAMEPAD] raw_axes={gamepad.raw_axes}, raw_buttons={gamepad.raw_buttons}, "
                    f"held={gamepad.held_names}, events={sorted(events)}, "
                    f"cmd[vx,vy,wz]={np.round(command, 3).tolist()}",
                    flush=True,
                )
                last_print = now
            if "exit" in events:
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

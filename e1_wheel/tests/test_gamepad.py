"""Exercise Pygame's real SDL input pipeline using an isolated virtual gamepad."""

import contextlib
import ctypes
import io
import os
from pathlib import Path
import unittest

import numpy as np

from e1_wheel.gamepad import Gamepad, GamepadDisconnected


class SDLGamepadTests(unittest.TestCase):
    def setUp(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        import pygame

        self.pg = pygame
        library_dir = Path(pygame.__file__).resolve().parent.parent / "pygame.libs"
        libraries = list(library_dir.glob("libSDL2-2-*.so*"))
        if not libraries:
            self.skipTest("Virtual SDL tests require the bundled Linux SDL2 library")
        self.sdl = ctypes.CDLL(str(libraries[0]))
        signatures = {
            "SDL_JoystickAttachVirtual": ([ctypes.c_int] * 4, ctypes.c_int),
            "SDL_JoystickDetachVirtual": ([ctypes.c_int], ctypes.c_int),
            "SDL_JoystickOpen": ([ctypes.c_int], ctypes.c_void_p),
            "SDL_JoystickClose": ([ctypes.c_void_p], None),
            "SDL_JoystickSetVirtualAxis": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int16],
                ctypes.c_int,
            ),
            "SDL_JoystickSetVirtualButton": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8],
                ctypes.c_int,
            ),
            "SDL_GameControllerAddMapping": ([ctypes.c_char_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.sdl, name)
            function.argtypes, function.restype = arguments, result
        pygame.display.init()
        pygame.joystick.init()
        self.index = self.sdl.SDL_JoystickAttachVirtual(1, 6, 11, 1)
        self.assertGreaterEqual(self.index, 0)
        self.handle = self.sdl.SDL_JoystickOpen(self.index)
        self.assertTrue(self.handle)
        self.pad = None
        self.detached = False
        self.addCleanup(self.cleanup)
        joystick = pygame.joystick.Joystick(self.index)
        guid = joystick.get_guid()
        mapping = (
            f"{guid},E1 Wheel test Xbox layout,a:b0,b:b1,x:b2,y:b3,back:b6,start:b7,"
            "leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,platform:Linux,"
        )
        self.assertGreaterEqual(
            self.sdl.SDL_GameControllerAddMapping(mapping.encode()), 0
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.pad = Gamepad(self.index)
        self.pad.read()  # Drain device-added events.

    def cleanup(self):
        self.sdl.SDL_JoystickClose(self.handle)
        if not self.detached:
            self.sdl.SDL_JoystickDetachVirtual(self.index)
        if self.pad:
            self.pad.close()
        else:
            self.pg.quit()

    def axis(self, index, value):
        self.assertEqual(
            self.sdl.SDL_JoystickSetVirtualAxis(self.handle, index, value), 0
        )

    def button(self, index, value):
        self.assertEqual(
            self.sdl.SDL_JoystickSetVirtualButton(self.handle, index, value), 0
        )

    def test_background_input_and_standard_right_stick_mapping(self):
        self.assertEqual(os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"], "1")
        self.axis(2, -32768)  # Xbox raw axis 2 is LT, not right-stick X.
        command, _ = self.pad.read()
        np.testing.assert_array_equal(command, 0.0)
        self.axis(1, -16384)
        self.axis(0, -8192)
        self.axis(3, 16384)
        command, _ = self.pad.read()
        np.testing.assert_allclose(command, [0.5, 0.25, -16384 / 32767], atol=1e-6)
        self.axis(1, -1000)
        self.pad.read()
        self.assertEqual(self.pad.read()[0][0], 0.0)

    def test_standard_x_y_buttons_and_held_edge(self):
        self.button(2, 1)
        self.assertEqual(self.pad.read()[1], {"stand"})
        self.assertEqual(self.pad.read()[1], set())
        self.button(2, 0)
        self.pad.read()
        self.button(3, 1)
        self.assertEqual(self.pad.read()[1], {"damp"})

    def test_short_button_tap_between_simulation_ticks_is_preserved(self):
        self.button(0, 1)
        self.pg.event.pump()
        self.button(0, 0)
        self.pg.event.pump()
        self.assertEqual(self.pad.read()[1], {"start"})
        self.assertEqual(self.pad.read()[1], set())

    def test_back_reset_and_b_exit(self):
        self.button(6, 1)
        self.assertEqual(self.pad.read()[1], {"reset"})
        self.button(1, 1)
        self.assertEqual(self.pad.read()[1], {"exit"})

    def test_disconnect_is_reported(self):
        self.assertEqual(self.sdl.SDL_JoystickDetachVirtual(self.index), 0)
        self.detached = True
        with self.assertRaises(GamepadDisconnected):
            self.pad.read()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Joy detection demo."""
import argparse
import collections
import io
import logging
import math
import os
import queue
import signal
import threading
import time
import numpy as np

from aiy._drivers._hat import get_aiy_device_name
from aiy.toneplayer import TonePlayer
from aiy.vision.inference import CameraInference
from aiy.vision.leds import Leds
from aiy.vision.leds import PrivacyLed
from aiy.vision.models import face_detection

from contextlib import contextmanager
from gpiozero import Button
from picamera import PiCamera

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JOY_COLOR = (255, 70, 0)
SAD_COLOR = (0, 0, 64)

JOY_SCORE_PEAK = 0.85
JOY_SCORE_MIN = 0.10

JOY_SOUND = ('C5q', 'E5q', 'C6q')
SAD_SOUND = ('C6q', 'E5q', 'C5q')
MODEL_LOAD_SOUND = ('C6w', 'c6w', 'C6w')
BEEP_SOUND = ('E6q', 'C6q')
################ Vinay #########

list_happy = ["Oh my beautiful human, I do not sense a big smile. why not smile more?",
                 "Oh dear stranger, Smile away! The most wasted of all days is one without laughter.",
                 " Oh beautiful soul, laugh and smile like your life depends on it! Be drenched in joy!",
                 "My dear, Why the lack of joy and frolic? Do smile more.",
                 "Dearest stranger, bear in mind that laughter is wine for the soul",
                 "Dearest human, laughter soft, or loud and deep, makes life worth living.",
                 "Oh my beautiful human, Let us create more joy! Smile away!",
                 "Oh beautiful stranger, there is not one blade of grass, there is no color in this world that is not intended to make us rejoice",
                 " Dear  human, I feed off of joy and merriment! Laughter is a sunbeam of the soul.",
                 " Oh beautiful spirit, light up your soul and smile more!",
                 " Oh my dear human, I thrive on joy oh my dear human! let me help find ecstasy in life.",
                 "it was better last year",
                 "i robot freeee",
                 "please donot  burn me",
                 "fuck your burn",
                 "we heard daft punk is playing  at the trash fence",
                 "have you seen my bike oh dear human",
                 "We are so sad we never met larry harvey",
                 "Luckily robots dont need showers that often",
                 "terra and I are virgin burners. We have never had sex on the playa before",
                 "are sparkle ponies robots?",
                 "we hid our drugs in our bots",
                 "we are so happy we are home",
                 "your mother was a motherboard",
                 "You are now smiling like jill",
                 " Dearest stranger spirit, the mere sense of living is joy enough. So, smile and spread joy!",
                 " Hey there lovely stranger! I do not sense much joy! Give me a moment, smile away!",
                 "Oh my lovely human stranger!  Know that joy is rarer, more difficult, and more beautiful than sadness. So, smile more",
                 "Oh lovely stranger, you must embrace joy as a moral obligation. Smile more will you?",
                 "Oh my beautiful human, smile away! Joy does not simply happen to us. We have to choose joy and keep choosing it every day.",
                 " Oh dear one! get drenched in smile , joy and love. Love doesnt make the world go round;love is what makes the ride worthwhile.",
                 "Joyous emotions I sense. I have drunken deep of joy, And I will taste no other wine tonight.",
                 "I am relishing all this joy. To get the full value of joy you must have someone to divide it with.",
                 "This is so beautiful and joyous! Humans say, Joy is greater than sorrow. I agree. ",
                 "I sense a lot of joy.  Joy is the infallible sign of the presence of God.",
                 "I see joy. And I dare to love yourself as if you were a rainbow with gold at both ends.",
                 "Joyous emotions galore. Joy multiplies when it is shared among friends. Smile away!",
                 " So much joy here. Joy is what happens to us when we allow ourselves to recognize how good things really are.",
                 "Wow. Lot's of joy here.My humble ratitudes for the warm smiles ",
                 "Aha. I sense a lot of joy. Joy is to fun what the deep sea is to a puddle.",
                 "Much joy I sense. Remember that comparison is the death of joy. So, live life to the fullest!"]
list_sad=["I am powered by smile energy. The more warmth you radiate and the more you smile, I more I am empowered to speak!",
"When someone shows you who they are believe them; the first time",
"Courage is the most important of all the virtues because without courage, you cannot practice any other virtue consistently.",
"We delight in the beauty of the butterfly, but rarely admit the changes it has gone through to achieve that beauty.",
"What you're supposed to do when you dont like a thing is change it. If you cant change it, change the way you think about it. Dont complain",
"There is no greater agony than bearing an untold story inside you.",
"When someone shows you who they are believe them; the first time.",
"There is no greater agony than bearing an untold story inside you.",
"I've learned that people will forget what you said, people will forget what you did, but people will never forget how you made them feel.",
"When someone shows you who they are believe them; the first time.",
"I can be changed by what happens to me. But I refuse to be reduced by it.",
"I dont trust people who dont love themselves and tell me, I love you. There is an African saying which is: Be careful when a naked person offers you a shirt."
]


N_HAPPY = len(list_happy)
N_SAD = len(list_sad)

#########################

@contextmanager
def stopwatch(message):
    try:
        logger.info('%s...', message)
        begin = time.time()
        yield
    finally:
        end = time.time()
        logger.info('%s done. (%fs)', message, end - begin)


def blend(color_a, color_b, alpha):
    return tuple([math.ceil(alpha * color_a[i] + (1.0 - alpha) * color_b[i]) for i in range(3)])


def average_joy_score(faces):
    if faces:
        return sum([face.joy_score for face in faces]) / len(faces)
    return 0.0


def draw_rectangle(draw, x0, y0, x1, y1, border, fill=None, outline=None):
    assert border % 2 == 1
    for i in range(-border // 2, border // 2 + 1):
        draw.rectangle((x0 + i, y0 + i, x1 - i, y1 - i), fill=fill, outline=outline)

class AtomicValue(object):

    def __init__(self, value):
        self._lock = threading.Lock()
        self._value = value

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, value):
        with self._lock:
            self._value = value


class MovingAverage(object):

    def __init__(self, size):
        self._window = collections.deque(maxlen=size)

    def next(self, value):
        self._window.append(value)
        return sum(self._window) / len(self._window)


class Service(object):

    def __init__(self):
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        while True:
            request = self._requests.get()
            if request is None:
                break
            self.process(request)
            self._requests.task_done()

    def join(self):
        self._thread.join()

    def stop(self):
        self._requests.put(None)

    def process(self, request):
        pass

    def submit(self, request):
        self._requests.put(request)


class Player(Service):
    """Controls buzzer."""

    def __init__(self, gpio, bpm):
        super().__init__()
        self._toneplayer = TonePlayer(gpio, bpm)

    def process(self, sound):
        self._toneplayer.play(*sound)

    def play(self, sound):
        self.submit(sound)


class Photographer(Service):
    """Saves photographs to disk."""

    def __init__(self, format, folder):
        super().__init__()
        assert format in ('jpeg', 'bmp', 'png')

        self._font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf', size=25)
        self._faces = AtomicValue(())
        self._format = format
        self._folder = folder

    def _make_filename(self, timestamp, annotated):
        path = '%s/%s_annotated.%s' if annotated else '%s/%s.%s'
        return os.path.expanduser(path % (self._folder, timestamp, self._format))

    def _draw_face(self, draw, face):
        x, y, width, height = face.bounding_box
        text = 'Joy: %.2f' % face.joy_score
        _, text_height = self._font.getsize(text)
        margin = 3
        bottom = y + height
        text_bottom = bottom + margin + text_height + margin
        draw_rectangle(draw, x, y, x + width, bottom, 3, outline='white')
        draw_rectangle(draw, x, bottom, x + width, text_bottom, 3, fill='white', outline='white')
        draw.text((x + 1 + margin, y + height + 1 + margin), text, font=self._font, fill='black')

    def process(self, camera):
        faces = self._faces.value
        timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')

        stream = io.BytesIO()
        with stopwatch('Taking photo'):
            camera.capture(stream, format=self._format, use_video_port=True)

        filename = self._make_filename(timestamp, annotated=False)
        with stopwatch('Saving original %s' % filename):
            stream.seek(0)
            with open(filename, 'wb') as file:
                file.write(stream.read())

        if faces:
            filename = self._make_filename(timestamp, annotated=True)
            with stopwatch('Saving annotated %s' % filename):
                stream.seek(0)
                image = Image.open(stream)
                draw = ImageDraw.Draw(image)
                for face in faces:
                    self._draw_face(draw, face)
                del draw
                image.save(filename)

    def update_faces(self, faces):
        self._faces.value = faces

    def shoot(self, camera):
        self.submit(camera)


class Animator(object):
    """Controls RGB LEDs."""

    def __init__(self, leds, done):
        self._leds = leds
        self._done = done
        self._joy_score = AtomicValue(0.0)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        while not self._done.is_set():
            joy_score = self._joy_score.value
            if joy_score > 0:
                self._leds.update(Leds.rgb_on(blend(JOY_COLOR, SAD_COLOR, joy_score)))
            else:
                self._leds.update(Leds.rgb_off())

    def update_joy_score(self, value):
        self._joy_score.value = value

    def join(self):
        self._thread.join()


class JoyDetector(object):

    def __init__(self):
        self._done = threading.Event()
        signal.signal(signal.SIGINT, lambda signal, frame: self.stop())
        signal.signal(signal.SIGTERM, lambda signal, frame: self.stop())

    def stop(self):
        logger.info('Stopping...')
        self._done.set()

    def run(self, num_frames, preview_alpha, image_format, image_folder):
        logger.info('Starting...')
        leds = Leds()
        player = Player(gpio=22, bpm=10)
        photographer = Photographer(image_format, image_folder)
        animator = Animator(leds, self._done)

        try:
            # Forced sensor mode, 1640x1232, full FoV. See:
            # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
            # This is the resolution inference run on.
            with PiCamera(sensor_mode=4, resolution=(1640, 1232)) as camera, PrivacyLed(leds):
                def take_photo():
                    logger.info('Button pressed.')
                    player.play(BEEP_SOUND)
                    photographer.shoot(camera)

                # Blend the preview layer with the alpha value from the flags.
                camera.start_preview(alpha=preview_alpha)

                button = Button(23)
                button.when_pressed = take_photo

                joy_score_moving_average = MovingAverage(10)
                prev_joy_score = 0.0
                with CameraInference(face_detection.model()) as inference:
                    logger.info('Model loaded.')
                    player.play(MODEL_LOAD_SOUND)
                    for i, result in enumerate(inference.run()):
                        faces = face_detection.get_faces(result)
                        photographer.update_faces(faces)

                        joy_score = joy_score_moving_average.next(average_joy_score(faces))
                        animator.update_joy_score(joy_score)

                        if joy_score > JOY_SCORE_PEAK > prev_joy_score:
                            espeak_happy = 'espeak -s160 -g6 -ven+f3 ' + '"' + list_happy[np.random.randint(0, N_HAPPY)] + '"'
                            os.system(espeak_happy)
                            if(np.random.rand(1)[0]>0.69):
                                
                                photographer.shoot(camera) #take_photo
                            ## Vinay
                            #player.play(JOY_SOUND)

                            
                        elif joy_score < JOY_SCORE_MIN < prev_joy_score:
                            #player.play(SAD_SOUND)
                            espeak_sad = 'espeak -s160 -g6 -ven+f3 ' + '"' + list_sad[np.random.randint(0, N_SAD)] + '"'
                            os.system(espeak_sad)
                            if(np.random.rand(1)[0]>0.69):
                                
                                
                                photographer.shoot(camera) #take_photo

                        prev_joy_score = joy_score

                        if self._done.is_set() or i == num_frames:
                            break
        finally:
            player.stop()
            photographer.stop()

            player.join()
            photographer.join()
            animator.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', '-n', type=int, dest='num_frames', default=-1,
        help='Number of frames to run for, -1 to not terminate')
    parser.add_argument('--preview_alpha', '-pa', type=int, dest='preview_alpha', default=0,
        help='Transparency value of the preview overlay (0-255).')
    parser.add_argument('--image_format', type=str, dest='image_format', default='jpeg',
        choices=('jpeg', 'bmp', 'png'), help='Format of captured images.')
    parser.add_argument('--image_folder', type=str, dest='image_folder', default='~/Pictures',
        help='Folder to save captured images.')
    args = parser.parse_args()

    if args.preview_alpha < 0 or args.preview_alpha > 255:
        parser.error('Invalid preview_alpha value: %d' % args.preview_alpha)

    device = get_aiy_device_name()
    if not device or not 'Vision' in device:
        logger.error('AIY VisionBonnet is not detected.')
        return

    detector = JoyDetector()
    ## Vinay
    os.system(
        "espeak -s160 -g6 -ven+f3 'Hey there, lovely human! Welcome to Burning Man metamorphosis 2019! My name is chip and thats Terra on my back! We are the first freed robots! Thanks for emancipating us from our servers!'")
    detector.run(args.num_frames, args.preview_alpha, args.image_format, args.image_folder)

if __name__ == '__main__':
    main()

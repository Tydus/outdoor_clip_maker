from __future__ import annotations # This is required for "forward declartion", see also https://stackoverflow.com/a/55344418

import tqdm.auto as tqdm
from typing import Tuple, Dict, List
import subprocess

from util import *
from layers import *
from animations import *

class Clip(object):
    def __init__(
        self, screen_size: Tuple[int, int], fps: int, output_filename: str,
        center: Tuple[float, float], zoom: float,
        pois: Dict[str, PoI], paths: Dict[str, Path], animations: List[Animation],
        msaa: int=8,
        no_video: bool=False,
        keep_bmp: bool=False,
):
        self.screen_size = screen_size
        self.fps = fps
        
        self.canvas = FlattenLayer('tmp/' + output_filename + '_%05d.bmp', self.screen_size, zoom=zoom)

        bglayer   = self.canvas.add_new_layer(BackgroundLayer)
        pathlayer = self.canvas.add_new_layer(PathLayer, msaa=msaa)
        melayer   = self.canvas.add_new_layer(MeLayer, msaa=msaa)
        poilayer  = self.canvas.add_new_layer(PoiLayer, msaa=msaa)
        hudlayer  = self.canvas.add_new_layer(HUDLayer)
        imglayer  = self.canvas.add_new_layer(ImageAbsoluteLayer)
        
        self.canvas.set_center(center)
        melayer.set_me(center)
        
        [poilayer.add_poi(i) for i in pois.values()]
        [pathlayer.add_path(i) for i in paths.values()]
        
        self._animation_queue = self._process_animations(animations)
        
        self.total_length = max(
            k + v._length
            for k, v in 
            self._animation_queue
        )
        
        print("Total length: %d frame (%d s)" % (self.total_length, self.total_length // self.fps))
        
        subprocess.run(f'rm -f tmp/{output_filename}_*.bmp', shell=True, check=True)
        
        self._run_till_complete()
        
        if no_video: return
    
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(self.fps),
            '-i', 'tmp/' + output_filename + '_%05d.bmp',
            '-vf', 'format=yuv420p',
            '-crf', '31',
            output_filename,
        ], check=True)
        
        if not keep_bmp:
            subprocess.run(f'rm -f tmp/{output_filename}_*.bmp', shell=True, check=True)

        
    def _process_animations(self, animations: List[Animation]):
        ret = []
        current_time = 0
        total_time = 0
        
        for k, v in animations:
            # Set clip first so we have v._length available.
            v.set_clip(self)
            
            # Process different time format.
            if type(k) == int:
                t = k
            else:
                sgn = 1
                k = k.strip()
                if k[0] == '+': # '+00:01' means 1 second from the start time of the last animation.
                    t = current_time
                    k = k[1:]
                elif k[0] == '|': # '|00:01' means 1 second from the finalize time of the existing animations. 
                    t = total_time
                    k = k[1:]
                elif k[0] == '-': # '-00:01' means 1 second before the finalize time of the existing animations.
                    t = total_time
                    k = k[1:]
                    sgn = -1
                else:
                    t = 0
                
                s = time.strptime(':'.join(('00:00:' + k).split(':')[-3:]), '%H:%M:%S')
                t += sgn * self.fps * (((s.tm_hour * 60) + s.tm_min * 60) + s.tm_sec)
            
            ret.append((t, v))
            
            current_time = t
            total_time = max(total_time, current_time + v._length)
        
        ret.sort(key=lambda x: x[0])
        
        for k, v in ret:
            print(k // self.fps, v._length // self.fps, v.__class__.__name__)
        
        return ret

    def _run_till_complete(self):
        frame = 0
        
        queue = self._animation_queue
        ongoing = []
        done = []
        
        for frame in tqdm.trange(self.total_length):

            # Phase 1:
            # Check the queue and pop all animations which starts from the current frame to ongoing list.
            while queue and queue[0][0] == frame:
                ani = iter(queue.pop(0)[1])
                print(ani.__class__.__name__)
                if ani._length != 0:
                    # Handle 0-length (oneshot) animations: only prepare() it and never add it to ongoing.
                    ongoing.append(ani)
            
            # Phase 2: fire every animation in the ongoing list and keep the unfinished ones.
            old_ongoing, ongoing = ongoing, []
            for a in old_ongoing:
                try:
                    next(a)
                    ongoing.append(a)
                except StopIteration:
                    done.append(a)
            
            # Phase 3: render the frame!
            self.canvas.save_next_frame()
            
            # Phase 4: teardown all animations in the done list.
            for a in done: a.finalize()
            done = []

        assert queue == ongoing == [], 'queue=%s, ongoing=%s' % (queue, ongoing)
        
__all__ = [
    'Clip', 
    'PanZoomTo', 'MoveAlong', 'ShowImage', 'ShowVideo', 'CustomAni',
    'PoI', 'Path',
    'POI_ICONS',
]

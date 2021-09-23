import tqdm
from typing import Tuple, Dict, List

from util import *
from layers import *
from animations import *

class Movie(object):
    def __init__(self, screen_size: Tuple[int, int], fps: int, center: Tuple[float, float], zoom: float, pois: Dict[str, PoI], paths: Dict[str, Path], animations: List[Animation], msaa: int=8, filename_prefix: str=''):
        self.screen_size = screen_size
        self.fps = fps
        
        self.canvas = FlattenLayer(filename_prefix + '%04d.bmp', self.screen_size, zoom=zoom)

        bglayer   = self.canvas.add_new_layer(BackgroundLayer)
        pathlayer = self.canvas.add_new_layer(PathLayer, msaa=msaa)
        melayer   = self.canvas.add_new_layer(MeLayer, msaa=msaa)
        poilayer  = self.canvas.add_new_layer(PoiLayer, msaa=msaa)
        hudlayer  = self.canvas.add_new_layer(HUDLayer)
        #imglayer  = timeline.add_new_layer(ImageAbsoluteLayer)
        
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
        
        self._run_till_complete()
        
    def _process_animations(self, animations: List[Animation]):
        ret = []
        current_time = 0
        
        for k, v in animations:
            # Process different time format.
            if type(k) == int:
                t = k
            else:
                k = k.strip()
                if k[0] == '+': # '+00:01'
                    t = current_time
                    k = k[1:]
                else:
                    t = 0
                
                s = time.strptime(':'.join(('00:00:' + k).split(':')[-3:]), '%H:%M:%S')
                t += self.fps * (((s.tm_hour * 60) + s.tm_min * 60) + s.tm_sec)
            
            v.set_movie(self)
            print(t // self.fps, v._length // self.fps, type(v))
            ret.append((t, v))
            
            current_time = t
        
        ret.sort(key=lambda x: x[0])
        
        return ret

    def _run_till_complete(self):
        frame = 0
        
        queue = self._animation_queue
        ongoing = []
        
        
        for frame in tqdm.trange(self.total_length):

            # Phase 1:
            # Check the queue and pop all animations which starts from the current frame to ongoing list.
            while queue and queue[0][0] == frame:
                ongoing.append(iter(queue.pop(0)[1]))
            
            # Phase 2: fire every animation in the ongoing list and keep the unfinished ones.
            new_ongoing = []
            for a in ongoing:
                try:
                    next(a)
                    new_ongoing.append(a)
                except StopIteration:
                    pass
            ongoing = new_ongoing
            
            # Phase 3: render the frame!
            self.canvas.save_next_frame()

        assert queue == ongoing == []
            
__all__ = [
    'Movie', 
    'PanZoomTo', 'MoveAlong',
    'PoI', 'Path',
    'POI_ICONS',
]
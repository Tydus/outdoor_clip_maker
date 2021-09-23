import time
import math

from util import *
from layers import *

class Animation(object):
    def __init__(self, length: Union[int, str], *args, **kwargs):
        if type(length) == str:
            s = time.strptime(':'.join(('00:00:' + length).split(':')[-3:]), '%H:%M:%S')
            self._length_s = ((s.tm_hour * 60) + s.tm_min * 60) + s.tm_sec
            self._length = None
        else:
            self._length = length

        self._frame = 0
        self._movie = None
        
    def set_movie(self, movie: Layer):
        self._movie = movie
        
        # Defer length setting until here because fps is TBD when __init__().
        self._length = self._length or self._movie.fps * self._length_s
        
    def __iter__(self):
        assert type(self._movie.canvas) == FlattenLayer
        self.prepare()
        return self
    
    def __next__(self):
        if self._frame == self._length:
            raise StopIteration
        self.next_frame()
        self._frame += 1
    
    def prepare(self):
        pass
    
    def next_frame(self):
        raise NotImplementedError

class PanZoomTo(Animation):
    def __init__(self, length: Union[int, str], center: Tuple[float, float]=None, zoom: float=None, bounding_box: tuple=None):
        Animation.__init__(self, length)
        
        if bounding_box:
            assert center == zoom == None, 'bounding_box and center/zoom should be mutal exclusive.'
            
            self._target_center, self._target_zoom = bounding_box2center_zoom(bounding_box)
            
        else:
            assert bounding_box == None, 'bounding_box and center/zoom should be mutal exclusive.'
            
            self._target_center = center
            self._target_zoom = zoom
        
    def prepare(self):
        self._start_center = self._movie.canvas._center
        self._start_zoom   = self._movie.canvas._zoom
        
        if self._target_center:
            self._step_center = (
                (self._target_center[0] - self._start_center[0]) / self._length,
                (self._target_center[1] - self._start_center[1]) / self._length,
            )
        if self._target_zoom:
            self._step_zoom = (self._target_zoom - self._start_zoom) / self._length
        
        return self
    
    def next_frame(self):
        frame = self._frame + 1 # 1 ~ n
        
        if self._target_center:
            self._movie.canvas.set_center((
                frame * self._step_center[0] + self._start_center[0],
                frame * self._step_center[1] + self._start_center[1],
            ))

        if self._target_zoom:
            self._movie.canvas.set_zoom(
                frame * self._step_zoom + self._start_zoom,
            )
            
class MoveAlong(Animation):
    def __init__(self, length: Union[int, str], path: Path, hud_roads: str=''):
        Animation.__init__(self, length)
        self._path = path
        self._road_icons = [google_road_icon(i).array for i in hud_roads.split(' ')] if hud_roads else []
        
    def prepare(self):
        self._path.generate_smoothed(self._length)
        self._traveled_distance = 0.
        
        self._speed = self._path.total_length / self._length # m / f

        self._melayer = self._movie.canvas.find_first_layer_by_type(MeLayer)
        self._hudlayer = self._movie.canvas.find_first_layer_by_type(HUDLayer)
    
    def next_frame(self):
        # TODO: check off-by-one.
        frame = self._frame
        
        path = self._path
        swp = path.smoothed_waypoints
        
        self._movie.canvas.set_center(swp[frame])
        
        self._traveled_distance += self._speed

        p1 = swp[max(0, frame - 1)]
        p2 = swp[min(frame + 1, len(swp) - 1)]
        
        self._melayer.set_me(
            path.get_target_coordinate(self._traveled_distance),
            math.atan2(p2[1] - p1[1], p2[0] - p1[0]) / math.pi * 180,
        )
        
        # Workaround: hide icon and HUD on the last frame
        if self._frame + 1 == self._length:
            self._melayer.set_icon(None)
            self._hudlayer.set_road_icons(*self._road_icons)
        elif self._frame == 0:
            self._melayer.set_icon(self._path.icon)
            self._hudlayer.set_road_icons()
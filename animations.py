from __future__ import annotations # This is required for "forward declartion", see also https://stackoverflow.com/a/55344418

import time
import math
import enum
from PIL import Image

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
        self._clip = None
        
    def set_clip(self, clip: Layer):
        self._clip = clip
        
        # Defer length setting until here because fps is TBD when __init__().
        self._length = self._length or self._clip.fps * self._length_s
        
    def __iter__(self):
        assert type(self._clip.canvas) == FlattenLayer
        self.prepare()
        return self
    
    def __next__(self):
        self.next_frame()
        self._frame += 1
        if self._frame == self._length:
            raise StopIteration
    
    def prepare(self):
        pass
    
    def next_frame(self):
        raise NotImplementedError

class PanZoomTo(Animation):
    def __init__(self, length: Union[int, str], center: Tuple[float, float]=None, zoom: float=None, bounding_box: tuple=None):
        Animation.__init__(self, length)
        
        self._target_bounding_box = bounding_box
        self._target_center = center
        self._target_zoom = zoom
        
        if bounding_box:
            assert center == zoom == None, 'bounding_box and center/zoom should be mutal exclusive.'
        else:
            assert bounding_box == None, 'bounding_box and center/zoom should be mutal exclusive.'

            
    def prepare(self):
        if self._target_bounding_box:
            self._target_center, self._target_zoom = bb2center_zoom(self._target_bounding_box, self._clip.canvas.screen_size)
        
        self._start_center = self._clip.canvas._center
        self._start_zoom   = self._clip.canvas._zoom
        
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
            self._clip.canvas.set_center((
                frame * self._step_center[0] + self._start_center[0],
                frame * self._step_center[1] + self._start_center[1],
            ))

        if self._target_zoom:
            self._clip.canvas.set_zoom(
                frame * self._step_zoom + self._start_zoom,
            )
            
class MoveAlong(Animation):
    def __init__(self, length: Union[int, str], path: Path, hud_roads: str='', angle: int=None, flip: bool=False, dont_drift: bool=True):
        Animation.__init__(self, length)
        self._path = path
        self._road_icons = [google_road_icon(i).array for i in hud_roads.split(' ')] if hud_roads else []
        self._fixed_angle = angle and int(angle)
        self._flip = flip
        self._dont_drift = dont_drift
        
    def prepare(self):
        self._path.generate_smoothed(self._length)
        self._traveled_distance = 0.
        
        self._speed = self._path.total_length / self._length # m / f

        self._melayer = self._clip.canvas.find_first_layer_by_type(MeLayer)
        self._hudlayer = self._clip.canvas.find_first_layer_by_type(HUDLayer)
        self._pathlayer = self._clip.canvas.find_first_layer_by_type(PathLayer)
    
    def next_frame(self):
        # TODO: check off-by-one.
        frame = self._frame
        
        
        path = self._path
        swp = path.smoothed_waypoints
        
        self._clip.canvas.set_center(swp[frame])
        
        self._traveled_distance += self._speed
        
        d = self._traveled_distance
        
        #import pdb; pdb.set_trace()
        
        # Just an empirical formula.
        # z15=>256
        # z23=>1
        step_width = 2 ** (23 - self._clip.canvas._zoom) # m
        
        if self._dont_drift:
            p1 = path.get_target_coordinate(max(0, d - step_width))
            p2 = path.get_target_coordinate(min(d + step_width, path.total_length))
            self._melayer.set_pt(p1, p2)
            p1 = wgs2ratio(p1)
            p2 = wgs2ratio(p2)

        else:
            p1 = swp[max(0, frame - 1)]
            p2 = swp[min(frame + 1, len(swp) - 1)]
        
        if self._fixed_angle != None: # DO NOT use 'or' here (since fixed_angle may be 0).
            heading = self._fixed_angle 
        else:
            heading = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) / math.pi * 180
        
        self._melayer.set_me(path.get_target_coordinate(d), heading, self._flip)
        
        self._hudlayer.set_hud_text('%.2f / %.2f km' % (d / 1000, path.total_length / 1000))
        
        # Workaround: hide icon and HUD on the last frame
        if self._frame + 1 == self._length:
            self._melayer.set_icon(None)
            self._hudlayer.set_road_icons()
            self._hudlayer.set_hud_text('')
            self._pathlayer.set_highlight()
            self._melayer.set_pt(None, None)
        elif self._frame == 0:
            self._melayer.set_icon(self._path.icon)
            self._hudlayer.set_road_icons(*self._road_icons)
            self._pathlayer.set_highlight(self._path)
            
            
class ShowImage(Animation):
    def __init__(self, length: Union[int, str], arr: np.ndarray, coord: Tuple[int, int]):
        Animation.__init__(self, length)
        self.arr = arr
        self.coord = coord
        
    def prepare(self):
        self._imglayer = self._clip.canvas.find_first_layer_by_type(ImageAbsoluteLayer)
        assert self.arr.shape[1] + self.coord[0] <= self._clip.screen_size[0]
        assert self.arr.shape[0] + self.coord[1] <= self._clip.screen_size[1]

    def next_frame(self):
        # Workaround: hide image on the last frame
        if self._frame + 1 == self._length:
            self._imglayer.del_image(self.arr)
        elif self._frame == 0:
            self._imglayer.add_image(self.coord, self.arr)
        
            
class ShowVideo(Animation):
    def __init__(self, length: Union[int, str], video_dir: str, coord: Tuple[int, int]):
        Animation.__init__(self, length)
        
        self.video_dir = video_dir
        self._image_filenames = sorted([i for i in os.listdir(video_dir) if i.endswith('.bmp')])
        self.coord = coord
        
    def prepare(self):
        assert self._length == len(self._image_filenames), 'Video length should match animation length'
                                
        self._imglayer = self._clip.canvas.find_first_layer_by_type(ImageAbsoluteLayer)

    def next_frame(self):
        self._imglayer.clear()
            
        # Workaround: hide image on the last frame
        if self._frame + 1 != self._length:
            im = Image.open(self.video_dir + '/' + self._image_filenames[self._frame])
            arr = np.asarray(im, dtype='float32')/255.
            self._imglayer.add_image(self.coord, arr)
        
class CustomAni(Animation):
    ''' Custom Animation '''
    
    def __init__(self, length: Union[int, str], prepare_cb: Callable[[Clip], None]=None, frame_cb: Callable[[Clip], None]=None):
        Animation.__init__(self, length)
        self._prepare_cb = prepare_cb or (lambda clip: None)
        self._frame_cb = frame_cb or (lambda clip: None)
        
    def prepare(self): self._prepare_cb(self._clip)
        
    def next_frame(self): self._frame_cb(self._clip)

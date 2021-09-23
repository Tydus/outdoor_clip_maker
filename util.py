from __future__ import annotations # This is required for "forward declartion", see also https://stackoverflow.com/a/55344418

import numpy
import math
import cv2
import os.path
import requests
import base64
import enum
import numpy as np
import numba
import json
import bisect
import scipy.interpolate
import itertools
import hashlib
from PIL import Image
from typing import Tuple, List, Union, Type
from dataclasses import dataclass, field
import scipy.ndimage.interpolation
import tqdm
import pickle


URL_TEMPLATE_TEMPLATE = 'https://maps.googleapis.com/maps/vt?pb=!1m5!1m4!1i%%d!2i%%d!3i%%d!4i256!2m3!1e0!2sm!3i0!3m17!2sen-US!3sJP!5e18!12m4!1e68!2m2!1sset!2sRoadmap!12m3!1e37!2m1!1ssmartmaps!12m4!1e26!2m2!1sstyles!2z%s!4e0!5m1!5f%d'

DPI = 2
#STYLE = 'p.v:on,s.t:19|p.v:off,s.t:5|p.v:on,s.t:81|p.v:on,s.t:81|s.e:l|p.v:off,s.t:2|p.v:off,s.t:3|s.e:l|p.v:off,s.t:65|s.e:l.t|p.v:off,s.t:66|s.e:l.t|p.v:off,s.t:6|p.v:on'
STYLE = 's.t:2|p.v:off,s.t:40|s.e:g|p.v:simplified,s.t:40|s.e:l.i|p.v:off,s.t:82|s.e:l|p.v:off,s.t:49|s.e:g.s|p.v:off,s.t:49|s.e:g.f|p.v:on|p.c:#ffffffff,s.t:50|s.e:g.s|p.v:off,s.t:51|s.e:g.s|p.v:off,s.t:81|p.v:on,s.t:20|p.v:off,s.e:g.s|p.v:simplified,s.t:19|s.e:g|p.v:off,s.t:49|s.e:l.i|p.v:off,s.t:81|s.e:g.f|p.v:simplified,s.t:4|s.e:l|p.v:off,s.t:1059|s.e:l.i|p.v:off'

GOOGLE_POI_ICON_TEMPLATE = 'https://www.google.com/maps/vt/icon/name=assets/icons/poi/tactile/pinlet_outline_v4-2-medium.png,assets/icons/poi/tactile/pinlet_v4-2-medium.png,assets/icons/poi/quantum/pinlet/%s_pinlet-2-medium.png&highlight=%s,%s,ffffff?scale=%d'

def encode_style(style):
    return base64.b64encode(style.encode()).decode().rstrip('=')

URL_TEMPLATE = URL_TEMPLATE_TEMPLATE % (encode_style(STYLE), DPI)

TILE_SIZE = 256 * DPI
CACHE_DIR = 'tiles/%s' % hashlib.md5(URL_TEMPLATE.encode()).hexdigest()[:16]
print(CACHE_DIR)
#!mkdir -p {CACHE_DIR}

http = requests.Session()

class ImageAnchor(enum.Enum):
    NW = (-1, -1)
    N  = ( 0, -1)
    NE = ( 1, -1)
    W  = (-1,  0)
    C  = ( 0,  0)
    E  = ( 1,  0)
    SW = (-1,  1)
    S  = ( 0,  1)
    SE = ( 1,  1)
    

class ImageClip(object):
    def __init__(self, filename_or_arr: Union[np.ndarray, str], anchor: ImageAnchor=ImageAnchor.NW, zoom: float=1.):
        self._anchor = anchor
        
        if type(filename_or_arr) == np.ndarray:
            # Reuse existing array (use "ImageClip(arr.copy())" if you need a copy)
            self.array = filename_or_arr
            return
        
        im = Image.open(open(filename_or_arr, 'rb'))
        if 'A' in im.mode:
            im = im.convert('RGBA')
        elif im.mode == 'P' and 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')

        self.array = numpy.asarray(im, dtype='float32')/255.
            
        if zoom != 1.:
            self.array = cv2.resize(self.array, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
            

    def paste_onto(self, canvas: np.ndarray, coordinate: Tuple[int, int]):
        
        gravity = (np.array((1, 1)) + self._anchor.value) / 2
        
        lt_coord = coordinate - (gravity * self.array.shape[:2][::-1]).astype('int')
        rb_coord = lt_coord + self.array.shape[:2][::-1]
        
        if np.any(lt_coord < 0) or np.any(rb_coord > canvas.shape[:2][::-1]):
            # We are out of bound. Don't draw anything.
            # TODO: implement partial drawing.
            return False
        
        slice_ = (
            slice(lt_coord[1], rb_coord[1]),
            slice(lt_coord[0], rb_coord[0]),
            Ellipsis,
        )
        
        canvas[slice_] = alpha_composite(canvas[slice_], self.array)
        
        return True

def test_imgclip():
    s = np.array(bg.array.shape)
    s[2] = 4

    cg = ImageClip(cv2.circle(np.zeros(s), (1024, 512), 20, (1, 0, 1, 0.5), -1))

    bg = ImageClip('88.png')

    fg = ImageClip('assets/pin.png', anchor=ImageAnchor.S)

    bg.array[512, :, ...] = 0
    bg.array[:, 1024, ...] = 0

    fg.paste_onto(cg.array, (1024, 512))
    cg.paste_onto(bg.array, (0, 0))

    Show(bg.array)

def google_poi_icon(name, color1, color2, scale=4):
    #assert name in GOOGLE_POI_ICON_VALID_LIST

    url = GOOGLE_POI_ICON_TEMPLATE % (name, color1, color2, min(scale, 4))
    
    res = http.get(url, stream=True)
    im = Image.open(res.raw).convert('RGBA')
        
    arr = numpy.asarray(im, dtype='float32')/255.
    
    # handle scale > 4
    if scale > 4:
        arr = cv2.resize(arr, (0, 0), fx=scale / 4, fy=scale / 4)
        
    return ImageClip(arr, ImageAnchor.S)

def test_google_icon():
    Show(np.hstack([
        google_poi_icon('airport', '1967d2', '1a73e8').array,
        google_poi_icon('bank_jp', '5c6bc0', '7986cb').array,
        google_poi_icon('bar', 'ea8600', 'f29900').array,
        google_poi_icon('bridge', '607d8b', '78909c').array,
        google_poi_icon('cafe', 'ea8600', 'f29900').array,
        google_poi_icon('camera', '129eaf', '12b5cb').array,
        google_poi_icon('camping', '1e8e3e', '34a853').array,
        google_poi_icon('car_rental', '5c6bc0', '7986cb').array,
        google_poi_icon('cemetery_jp', '1e8e3e', '34a853').array,
        google_poi_icon('city_office_jp', '607d8b', '78909c').array,
        google_poi_icon('civil_office_jp', '607d8b', '78909c').array,
        google_poi_icon('constellation_star', 'e2ab20', 'fbc02d').array,
        google_poi_icon('dolphin', '129eaf', '12b5cb').array,
    ]))
    Show(np.hstack([
        google_poi_icon('dot', '1e8e3e', '34a853').array,
        google_poi_icon('dot', '607d8b', '78909c').array,
        google_poi_icon('dot', '607d8b', '7b9eb0').array,
        google_poi_icon('event_venue', '129eaf', '12b5cb').array,
        google_poi_icon('ferriswheel', '129eaf', '12b5cb').array,
        google_poi_icon('flower', '1e8e3e', '34a853').array,
        google_poi_icon('golf', '1e8e3e', '34a853').array,
        google_poi_icon('heart', 'b64371', 'fa507d').array,
        google_poi_icon('hiking', '1e8e3e', '34a853').array,
        google_poi_icon('home', '1a73e8', '4285f4').array,
        google_poi_icon('hotspring', '1e8e3e', '34a853').array,
        google_poi_icon('library', '607d8b', '78909c').array,
        google_poi_icon('lodging', 'ec407a', 'f06292').array,
    ]))
    Show(np.hstack([
        google_poi_icon('mountain', '1e8e3e', '34a853').array,
        google_poi_icon('movie', '129eaf', '12b5cb').array,
        google_poi_icon('museum_jp', '129eaf', '12b5cb').array,
        google_poi_icon('museum_jp', '80868b', '9aa0a6').array,
        google_poi_icon('nickname', '1a73e8', '4285f4').array,
        google_poi_icon('note', '129eaf', '12b5cb').array,
        google_poi_icon('palette', '129eaf', '12b5cb').array,
        google_poi_icon('parking', '5c6bc0', '7986cb').array,
        google_poi_icon('paw', '1e8e3e', '34a853').array,
        google_poi_icon('police_jp', '607d8b', '78909c').array,
        google_poi_icon('postoffice_jp', '607d8b', '78909c').array,
        google_poi_icon('relic_jp', '129eaf', '12b5cb').array,
        google_poi_icon('relic_jp', '80868b', '9aa0a6').array,
    ]))
    Show(np.hstack([
        google_poi_icon('resort', '1e8e3e', '34a853').array,
        google_poi_icon('restaurant', 'ea8600', 'f29900').array,
        google_poi_icon('school_cn_jp', '607d8b', '78909c').array,
        google_poi_icon('shoppingbag', '4285f4', '5491f5').array,
        google_poi_icon('shoppingbag', 'c5221f', 'ea4335').array,
        google_poi_icon('shoppingcart', '4285f4', '5491f5').array,
        google_poi_icon('stadium', '1e8e3e', '34a853').array,
        google_poi_icon('theater', '129eaf', '12b5cb').array,
        google_poi_icon('tree', '1e8e3e', '34a853').array,
        google_poi_icon('work', '1a73e8', '4285f4').array,
        google_poi_icon('worship_buddhist', '607d8b', '78909c').array,
        google_poi_icon('worship_islam', '607d8b', '78909c').array,
        google_poi_icon('worship_shinto', '607d8b', '78909c').array,
    ]))


POI_ICONS = {
    'Auto parts store': google_poi_icon('car_rental', '5c6bc0', '7986cb', 16), 
    'Bridge': google_poi_icon('bridge', '607d8b', '78909c', 16),
    'Convenience store': google_poi_icon('convenience', '4285f4', '5491f5', 16),
    'Gas station': google_poi_icon('gas', '5c6bc0', '7986cb', 16), 
    'Hotel': google_poi_icon('lodging', 'ec407a', 'f06292', 16),
    'Lodging': google_poi_icon('home', '1a73e8', '4285f4', 16), # Home
    'Museum': google_poi_icon('museum_jp', '80868b', '9aa0a6', 16),
    'Onsen': google_poi_icon('hotspring', '1e8e3e', '34a853', 16),
    'Restaurant': google_poi_icon('restaurant', 'ea8600', 'f29900', 16),
    'Rest stop': google_poi_icon('parking', '5c6bc0', '7986cb', 16), # Parking
    'Seaport': google_poi_icon('boating', '1e8e3e', '34a853', 16),
    'Shopping mall': google_poi_icon('shoppingbag', 'c5221f', 'ea4335', 16),
    'Sightseeing spot': google_poi_icon('tree', '1e8e3e', '34a853', 16),
    'Supermarket': google_poi_icon('shoppingcart', '4285f4', '5491f5', 16),
    'Tourist attraction': google_poi_icon('camera', '129eaf', '12b5cb', 16),
    'Train station': google_poi_icon('dot', 'c5221f', 'ea4335', 16), # Red dot
    'Video arcade': google_poi_icon('dice', 'c5221f', 'ea4335', 16),
    '_default': google_poi_icon('dot', '607d8b', '78909c', 16),
}


#DEFAULT_POI_ICON = ImageClip('assets/pin.png', anchor=ImageAnchor.S)
DEFAULT_POI_ICON = google_poi_icon('dot', '607d8b', '78909c', 16)

def wgs2px_old(zoom: float, lat_lon: Tuple[float, float], float_: bool=False) -> Tuple[int, int]:
    lat, lon = lat_lon
    lambda_ = lon / 180 * math.pi
    phi = lat / 180 * math.pi
    k = TILE_SIZE / (2 * math.pi) * (2 ** zoom)
    
    x = lambda_ + math.pi
    y = math.pi - math.log(math.tan(math.pi / 4 + phi / 2))
    
    if float_:
        return (x * k), (y * k)
    else:
        return int(x * k), int(y * k)

def wgs2ratio(lat_lon: Tuple[float, float]) -> Tuple[float, float]:
    lat, lon = lat_lon
    lambda_ = lon / 180 * math.pi
    phi = lat / 180 * math.pi
    tau = 2 * math.pi
    
    #x = (math.pi + lambda_) / tau
    x = 1 / 2 + lambda_ / tau
    #y = (math.pi - math.log(math.tan(math.pi / 4 + phi / 2))) / tau
    y = 1 / 2 - math.log(math.tan(math.pi / 4 + phi / 2)) / tau
    
    return x, y

def ratio2px(zoom: float, mercator_ratio: Tuple[float, float], float_: bool=False) -> tuple:
    k = TILE_SIZE * (2 ** zoom)
    f = float if float_ else int
    ret = f(mercator_ratio[0] * k), f(mercator_ratio[1] * k)
    return ret

def wgs2px(zoom: float, lat_lon: Tuple[float, float], float_: bool=False) -> tuple:
    ret = ratio2px(zoom, wgs2ratio(lat_lon), float_)
    
    return ret
    
def wgsratio2px_relative(zoom: float, target_ratio: Tuple[float, float], center: Tuple[float, float], canvas_size: Tuple[int, int], float_: bool=False) -> tuple:
    p1 = ratio2px(zoom, target_ratio, float_=True) 
    p0 = wgs2px(zoom, center, float_=True)
    
    return (
        canvas_size[0] / 2 + p1[0] - p0[0],
        canvas_size[1] / 2 + p1[1] - p0[1],
    ) if float_ else (
        int(canvas_size[0] / 2 + p1[0] - p0[0]),
        int(canvas_size[1] / 2 + p1[1] - p0[1]),
    )
    
def wgs2px_relative(zoom: float, target: Tuple[float, float], center: Tuple[float, float], canvas_size: Tuple[int, int], float_: bool=False) -> tuple:
    #ret = (
    #    np.array(canvas_size) / 2 + 
    #    wgs2px(zoom, lat_lon, float_=True) - 
    #    wgs2px(zoom, center, float_=True)
    #)
    #
    #return ret if float_ else ret.astype('int')
    
    p1 = wgs2px_old(zoom, target, float_=True) 
    p0 = wgs2px_old(zoom, center, float_=True)
    
    return (
        canvas_size[0] / 2 + p1[0] - p0[0],
        canvas_size[1] / 2 + p1[1] - p0[1],
    ) if float_ else (
        int(canvas_size[0] / 2 + p1[0] - p0[0]),
        int(canvas_size[1] / 2 + p1[1] - p0[1]),
    )

def px2wgs(zoom: float, x_y: Tuple[int, int]) -> Tuple[float, float]:
    k = TILE_SIZE / (2 * math.pi) * (2 ** zoom)
    
    x, y = x_y
    x, y = x / k, y / k
    
    lambda_ = x - math.pi
    phi = (math.atan(math.e ** (math.pi - y)) - math.pi / 4) * 2
    
    lon = lambda_ * 180 / math.pi
    lat = phi * 180 / math.pi
    
    return lat, lon

# Calculate center, zoom from bounding box corners.
# WARNING: may break in high-latitude region.
def get_bounding_box_center_zoom(bb: float, screen_size: Tuple[int, int]):
    lt, rb = bb

    bb_center = (
        (lt[0] + rb[0]) / 2,
        (lt[1] + rb[1]) / 2,
    )

    rlt = wgs2ratio(lt)
    rrb = wgs2ratio(rb)

    # zoom = math.log(ratio * TILE_SIZE / px) / math.log(2)

    rdelta = max(
        (rrb[0] - rlt[0]) * TILE_SIZE / screen_size[0],
        (rrb[1] - rlt[1]) * TILE_SIZE / screen_size[1],
    )

    # Set a safe margin (make zoom smaller)
    bb_zoom = math.log(rdelta * 0.9, base=2)

    return bb_center, bb_zoom


def get_tile_index(zoom: int, lat_lon: Tuple[float, float]) -> Tuple[int, int, int]:
    ret = wgs2px(zoom, lat_lon)
    return zoom, int(ret[0] / TILE_SIZE), int(ret[1] // TILE_SIZE)

def get_one_tile(zoom: int, x: int, y: int) -> np.ndarray:
    cache_filename = os.path.join(CACHE_DIR, '%d_%d_%d.bmp' % (zoom, x, y))
    
    if os.path.isfile(cache_filename):
        im = Image.open(open(cache_filename, 'rb'))
    else:
        url = URL_TEMPLATE % (zoom, x, y)
        print(zoom, x, y)
        res = http.get(url, stream=True)
        im = Image.open(res.raw).convert('RGB')
        im.save(cache_filename)
        
    a = np.asarray(im, dtype='float32')/255.
    return a

def get_tiles(zoom: int, center_latlon: Tuple[float, float], extra_tiles: Tuple[int, int] = (0, 0))  -> np.ndarray:
    extra_x, extra_y = extra_tiles
    
    _, center_x, center_y = get_tile_index(zoom, center_latlon)
    
    left_top_xy = np.asarray([center_x - extra_x, center_y - extra_y]) * TILE_SIZE
    
    arr = np.hstack([
        np.vstack([
            get_one_tile(zoom, x, y)
            for y in range(center_y - extra_y, center_y + extra_y + 1)
        ]) for x in range(center_x - extra_x, center_x + extra_x + 1)
    ])
    return left_top_xy, arr

#@numba.jit(nopython=True)
def alpha_composite(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    #assert fg.dtype == 'float32' and bg.dtype == 'float32'
    assert len(fg.shape) == 3 and fg.shape[2] in [3, 4]
    assert len(bg.shape) == 3 and bg.shape[2] in [3, 4]
    assert fg.shape[:2] == bg.shape[:2]
    
    if fg.shape[2] == 3: return fg
    
    Cb, ab = bg[..., :3], bg[..., 3:]
    Ca, aa = fg[..., :3], fg[..., 3:]
    
    if bg.shape[2] == 3: # BG does not contain alpha channel.
        #a0 = np.ones((bg.shape[0], bg.shape[1], 1), dtype='float32')
        #a = np.divide(aa, a0, out=np.zeros_like(a0), where=a0 > 1e-6)
          
        C0 = aa * Ca + (1 - aa) * Cb
    
        return C0
    else:
        a0 = 1 - (1 - aa) * (1 - ab)
    
        #a = np.where(a0 > 1e-6, aa / a0, np.zeros_like(a0))
        a = np.divide(aa, a0, out=np.zeros_like(a0), where=a0 > 1e-6)
    
        C0 = a * Ca + (1 - a) * Cb
    
        return np.concatenate((C0, a0), 2)
    
# TODO: hard-coded top-center (i.e. North), support different alignments.
def draw_text(
    arr: np.ndarray,
    pos: Tuple[int, int], 
    text: Union[str, List[str]], 
    font=cv2.FONT_HERSHEY_SIMPLEX, color: tuple=(0, 0, 0, 1), 
    size: int=2, width: int=0, margin: int=0,
):
    if type(text) == str: text = [text]
    if len(text) == 0: return arr

    text, *left = text

    # Default width to be same as size.
    width = width or size
    
    w, h = cv2.getTextSize(text, font, size, width)[0]

    arr = cv2.putText(arr, text, (pos[0] - w // 2, pos[1] + h), font, size, color, width)

    pos = pos[0], pos[1] + h + margin
    return draw_text(arr, pos, left, font, color, size, width, margin)
    
# alpha_composite(np.zeros((1, 1, 3), dtype='float32'), np.zeros((1, 1, 3), dtype='float32'))

#bg = np.random.rand(1080, 1920, 3).astype('float32')
#fg = np.random.rand(1080, 1920, 4).astype('float32')
#fg[..., 3] = np.clip(fg[..., 3] * 2 - 1, 0, 1)

#%timeit alpha_composite(bg, fg)

#%timeit wgs2px_relative(12.34, (35.124958, 135.954876), (34.932275, 135.790481), (1920 * 8, 1080 * 8))
#%timeit wgs2px_old(12.34, (35.124958, 135.954876))
#r__ = wgs2ratio((35.124958, 135.954876))
#%timeit ratio2px(12.34, r__)

def test_text():

    y = 1500
    x = 2000

    arr = np.ones((4000, 4000, 4), dtype='float32')
    arr[y, :, 0] = arr[:, x, 0] = 0

    arr = draw_text(arr, (x, y), ['Lorem ipsum dolor sit amet', 'consectetur adipiscing elit', 'Morbi fermentum tortor ligula', 'tempus commodo tellus aliquet sed' ] * 30, size=8, width=8, margin=50)

    Show(arr, factor=0.2)

#test_text()

@dataclass(frozen=True, eq=True)
class PoI(object):
    location: Tuple[float, float]
    name: str
    icon: ImageClip = field(default=DEFAULT_POI_ICON)
    
    def __hash__(self):
        lat = int((90  + self.location[0]) * 1e9)
        lon = int((180 + self.location[1]) * 1e9)
        return hash((lat << 32) + lon)
       
    def __eq__(lhs, rhs):
        return hash(lhs) == hash(rhs)

class Path(object):
    # TODO: Freeze it.
    
    def __init__(self, *vertices: List[Union[PoI, Tuple[float, float]]], icon: str='car'):
        self.icon = icon
        
        v = [tuple(i.location) if type(i) == PoI else tuple(i) for i in vertices]
        
        # Remove (at least adjacent) duplicates in vertices.
        # This is important because will not work, saying `Invalid inputs.'
        self.vertices = [v[0]]
        for i in range(1, len(v)):
            if v[i] != v[i - 1]:
                self.vertices.append(v[i])

        assert len(self.vertices) > 1, 'Empty vertices'
        
        # Length of each path segments.
        # L(P_i, P_i+1)
        self._lengths = [
            self._distance(end, start) 
            for start, end in zip(self.vertices[:-1], self.vertices[1:])
        ]
        
        # Accumulated lengths from 0-th point to i-th point.
        # L(P_0, P_i)
        self._accumulated_lengths = [0] + list(itertools.accumulate(self._lengths))
        
        # Total length
        # L(P_0, P_i-1)
        self.total_length = self._accumulated_lengths[-1]
        
        # Calculate B-spline parameters
        #import pdb; pdb.set_trace()
        self._tck = scipy.interpolate.splprep(list(zip(*self.vertices)), s=0.01)[0]
        
        # Calculate mercator ratio for each points
        self._mercator_ratios = list(map(wgs2ratio, self.vertices))

    def __repr__(self):
        return "Path(%s, icon='%s')" % (', '.join(map(str, self.vertices)), self.icon)
        
    def generate_smoothed(self, n):
        self.smoothed_waypoints = list(zip(*scipy.interpolate.splev(
            np.linspace(0, 1, n),
            self._tck,
        )))
        
    def _distance(self, P1: Tuple[float, float], P2: Tuple[float, float]):
        ''' Get distance between two points, in meters. '''
        lat1, lon1 = P1
        lat2, lon2 = P2
        
        p = math.pi / 180.
        lat1 *= p; lon1 *= p; lat2 *= p; lon2 *= p
        
        a = 1 - math.cos(lat2 - lat1) + math.cos(lat1) * math.cos(lat2) * (1 - math.cos(lon2 - lon1))
        return 2 * 6371000 * math.asin(math.sqrt(a / 2))

    def get_segment_id(self, meters_from_start: float):
        '''
        Return the segment ID (from zero).
        I.e. the segment between i-th vertex and i+1-th.
        '''
        # clip to total_length to avoid overrun.
        if meters_from_start > self.total_length:
            meters_from_start = self.total_length
            
        return bisect.bisect_right(self._accumulated_lengths, meters_from_start) - 1
    
    def get_target_coordinate(self, meters_from_start: float):
        assert meters_from_start >= 0
        # Avoid overrun.
        if meters_from_start >= self.total_length:
            return self.vertices[-1]
        
        segment_id = self.get_segment_id(meters_from_start)
        
        #L_total - L(P_0, P_id)
        meters_left = meters_from_start - self._accumulated_lengths[segment_id]
        assert meters_left >= 0
        
        start, end = [np.asarray(i) for i in self.vertices[segment_id:segment_id + 2]]
            
        segment_length = self._distance(start, end)
        
        return tuple(start + (end - start) * (meters_left / segment_length))
    
    def get_points(self, canvas_zoom: float, canvas_center: Tuple[float, float], canvas_size: Tuple[int, int]) -> np.ndarray:
        ''' Get an array of points for vertices, given zoom and center coordinate'''
        
        # Manually inline wgsratio2px_relative() and ratio2px() for speed.
        k = TILE_SIZE * (2 ** canvas_zoom)
        canvas_size_half = np.array(canvas_size) / 2 # (2, )
        p0 = wgs2px(canvas_zoom, canvas_center)      # (2, )
    
        p1s = np.asarray(self._mercator_ratios) * k  # (n, 2)
        return (p1s - p0 + canvas_size_half).astype('int')
    
    @staticmethod
    def get_bounding_box(*paths: List[Path]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        ''' Get the "bounding box" of the path(s) given. I.e. the max/min of the longitude/latitude. '''
        
        return (
            tuple(np.min(np.vstack([np.min(i.vertices, axis=0) for i in paths]), axis=0)),
            tuple(np.max(np.vstack([np.max(i.vertices, axis=0) for i in paths]), axis=0)),
        )
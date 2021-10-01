from __future__ import annotations # This is required for "forward declartion", see also https://stackoverflow.com/a/55344418

import numpy as np
from util import *

class Layer(object):
    def __init__(self, screen_size: Tuple[int, int], zoom: float):
        self.screen_size = screen_size
        self._frame = 0
        self._center = (0, 0)
        self._zoom = zoom
        self._frame_cache = None
        
    def set_center(self, center: Tuple[float, float]):
        self._center = center
        self._frame_cache = None

    def offset_center(self, delta: Tuple[float, float]):
        self.set_center((self._center[0] + delta[0], self._center[1] + delta[1]))
        
    def set_zoom(self, zoom: float):
        self._zoom = zoom
        self._frame_cache = None

    def offset_zoom(self, delta: float):
        self.set_zoom(self._zoom + delta)


class FlattenLayer(Layer):
    def __init__(self, filename_pattern: str, *args, **kwargs):
        Layer.__init__(self, *args, **kwargs)
        
        self.sub_layers = []
        self._filename_pattern = filename_pattern
        
    def add_new_layer(self, layer_type: Type[Layer], *args, **kwargs):
        layer = layer_type(screen_size=self.screen_size, zoom=self._zoom, *args, **kwargs)
        self.add_layer(layer)
        return layer
    
    def set_center(self, center: Tuple[float, float]):
        self._center = center
        self._frame_cache = None
        [ i.set_center(center) for i in self.sub_layers ]
        
    def set_zoom(self, zoom: float):
        self._zoom = zoom
        self._frame_cache = None
        
        [ i.set_zoom(zoom) for i in self.sub_layers ]
        
    def add_layer(self, layer: Layer, z_index: int=-1):
        assert layer not in self.sub_layers, "Layer already exists."
        if z_index != -1:
            self.sub_layers.insert(layer, z_index)
        else:
            self.sub_layers.append(layer)
        
    def get_layer_index(self, layer: Layer) -> int:
        if layer in self.sub_layers:
            return self.sub_layers.index(layer)
        else:
            return -1
        
    def find_first_layer_by_type(self, layer_type: Type[Layer]) -> Layer:
        for i in self.sub_layers:
            if type(i) == layer_type:
                return i
        return None
    
    def skip_next_frame(self):
        ''' Skip rendering next frame, just increase frame count. '''
        self._frame += 1
    
    def render_next_frame(self) -> np.ndarray:
        self._frame += 1
        
        arr = self.sub_layers[0].render_next_frame()
        
        for l in self.sub_layers[1:]:
            arr = alpha_composite(arr, l.render_next_frame())
            
        return arr
    
    def save_next_frame(self):
        arr = self.render_next_frame()
        Image.fromarray((arr * 255).astype('uint8')).save(self._filename_pattern % self._frame)
    
class BackgroundLayer(Layer):

    def render_next_frame(self) -> np.ndarray:
        ''' Retrieve next frame '''
        if self._frame_cache is not None:
            return self._frame_cache
    
        int_zoom = round(self._zoom)
        
        # Handle non-integer zoom. render an int_zoom and scale down.
        n = 2 ** (int_zoom - self._zoom)
        int_screen_size = np.ceil(np.asarray(self.screen_size) * n).astype('int')

        extra_tiles = np.ceil(int_screen_size / 2 / TILE_SIZE).astype('int')

        left_top, canvas = get_tiles(int_zoom, self._center, extra_tiles)

        center_coor = wgs2px(int_zoom, self._center) - left_top
        new_lt = (center_coor - int_screen_size / 2).astype('int')
        new_rb = new_lt + int_screen_size

        canvas = canvas[new_lt[1]:new_rb[1], new_lt[0]:new_rb[0], ...]

        # Handle non-integer zoom. Scale to target canvas size.
        #if abs(int_zoom - self._zoom) > 1e-6:
        if self.screen_size != canvas.shape[:2][::-1]:
            canvas = cv2.resize(canvas, self.screen_size, interpolation=cv2.INTER_AREA)

        self._frame += 1
        self._frame_cache = canvas
        return canvas

class PoiLayer(Layer):
    def __init__(self, *args, msaa: int=8, **kwargs):
        Layer.__init__(self, *args, **kwargs)
        self._msaa = msaa
        self._pois = set()
        self._name_visibility = 0 # default
        
    def add_poi(self, poi: PoI) -> PoI:
        self._pois.add(poi)
        self._frame_cache = None
        return poi
        
    def remove_poi(self, poi: PoI):
        self._pois.remove(poi)
        self._frame_cache = None
        
    def clear(self):
        self._pois = set()
        self._frame_cache = None

    def set_name_visibility(self, visibility: str):
        # So we can use 'self._name_visibility or poi.name_visibility' to control per poi if default(0) is set.
        self._name_visibility = {
            'all': 1,
            'default': 0,
            'none': -1,
        }[visibility]
        
        
    def render_next_frame(self) -> np.ndarray:
        ''' Retrieve next frame '''
        if self._frame_cache is not None:
            return self._frame_cache
    
        canvas = np.zeros((self._msaa * self.screen_size[1], self._msaa * self.screen_size[0], 4), dtype='float32') # RGBA
        
        for i in self._pois:
            pt = wgs2px_relative(
                self._zoom + bin(self._msaa)[2:].count('0'), 
                i.location, self._center,
                np.array(self.screen_size) * self._msaa,
            )
            # Draw pt
            i.icon.paste_onto(canvas, pt)
            
            # Draw name
            if (self._name_visibility or i.name_visibility) > 0:
                canvas = draw_text(canvas, pt, i.name.split('\\n'), size=int(1 * self._msaa), width=2 * self._msaa, color=(1, 0, 0, 1))
        
        if self._msaa > 1:
            canvas = cv2.resize(canvas, self.screen_size, interpolation=cv2.INTER_AREA)
        
        self._frame += 1
        self._frame_cache = canvas
        return canvas

class PathLayer(Layer):
    def __init__(self, *args,
        line_width: int=4, line_color: Tuple[float, float, float, float]=(0.5, 0.5, 0.5, 0.8),
        highlight_line_width:int=16, highlight_line_color: Tuple[float, float, float, float]=(0, 82/255., 145/255., 1),
        msaa: int=8, **kwargs,
    ):
        Layer.__init__(self, *args, **kwargs)
        self._msaa = msaa
        self._line_width = line_width
        self._line_color = line_color
        self._highlight_line_width = highlight_line_width
        self._highlight_line_color = highlight_line_color
        
        self._paths = set()
        self._highlight_paths = set()
        
    def add_path(self, path: Path)-> Path:
        self._paths.add(path)
        self._frame_cache = None
        return path
        
    def remove_path(self, path: Path):
        self._paths.remove(Path)
        self._frame_cache = None

    def set_highlight(self, *paths: List[Path]):
        self._highlight_paths = set(paths)
        self._frame_cache = None
        
    def clear(self):
        self._paths = set()
        self._frame_cache = None
    
    def render_next_frame(self) -> np.ndarray:
        ''' Retrieve next frame '''
        if self._frame_cache is not None:
            return self._frame_cache
    
        canvas = np.zeros((self._msaa * self.screen_size[1], self._msaa * self.screen_size[0], 4), dtype='float32') # RGBA
        
        # Draw non-highlighted first.
        for path in self._paths.difference(self._highlight_paths):
            
            pts = path.get_points(
                self._zoom + bin(self._msaa)[2:].count('0'),
                self._center,
                (self.screen_size[0] * self._msaa, self.screen_size[1] * self._msaa),
            ).reshape(-1, 1, 2)

            # 2nd arg of cv2.polylines is very weird... See also https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
            canvas = cv2.polylines(canvas, [pts], 0, self._line_color, thickness=self._line_width * self._msaa, lineType=cv2.LINE_AA)

                    
        # Draw highlighted last (on top of the others).
        for path in self._highlight_paths:
            
            pts = path.get_points(
                self._zoom + bin(self._msaa)[2:].count('0'),
                self._center,
                (self.screen_size[0] * self._msaa, self.screen_size[1] * self._msaa),
            ).reshape(-1, 1, 2)

            # 2nd arg of cv2.polylines is very weird... See also https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
            canvas = cv2.polylines(canvas, [pts], 0, self._highlight_line_color, thickness=self._highlight_line_width * self._msaa, lineType=cv2.LINE_AA)

            
        if self._msaa > 1:
            canvas = cv2.resize(canvas, self.screen_size, interpolation=cv2.INTER_AREA)
        
        self._frame += 1
        self._frame_cache = canvas
        return canvas
    
    
        
class MeLayer(Layer):
    ''' Display an `me' icon (person or transits like train, car, etc.) '''
    
    icon_cache_dir = 'assets/_icon_cache'
    icon_filename_template = 'assets/%s.png'
    
    def __init__(self, *args, msaa: int=8, **kwargs):
        Layer.__init__(self, *args, **kwargs)
        self._me = None
        self._msaa = msaa
        self.icon = None # Set to none to hide icon.
        self._p1 = self._p2 = None
    
    def set_me(self, me: Tuple[float, float], heading: float=0, flip: bool=False):
        self._me = me
        self._heading = int(heading)
        self._flip = flip
        self._frame_cache = None
    
    def set_icon(self, icon_name: str):
        self._frame_cache = None
        if icon_name == None:
            self.icon = None
            return

        scale = self._msaa // 4

        self.icon = (icon_name, scale)

        im = Image.open(MeLayer.icon_filename_template % icon_name).convert('RGBA')
        self._icon_im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.ANTIALIAS)

    def set_pt(self, p1, p2):
        self._p1 = p1
        self._p2 = p2
    
    def render_next_frame(self) -> np.ndarray:
        ''' Retrieve next frame '''
        if self._frame_cache is not None:
            return self._frame_cache
    
        canvas = np.zeros((self._msaa * self.screen_size[1], self._msaa * self.screen_size[0], 4), dtype='float32') # RGBA
        
        pt = wgs2px_relative(
            self._zoom + bin(self._msaa)[2:].count('0'), 
            self._me or self._center, self._center,
            np.array(self.screen_size) * self._msaa,
        )
        # Draw icon
        if self.icon != None:
            _path = os.path.join(MeLayer.icon_cache_dir, self.icon[0], str(self.icon[1]))
            os.makedirs(_path, mode=0o755, exist_ok=True)
            
            _filename = os.path.join(_path, '%d.png' % self._heading)
            
            if os.path.isfile(_filename):
                im = Image.open(_filename)
            else:
                im = self._icon_im.rotate(angle=self._heading, resample=Image.BILINEAR)
                im.save(_filename)
            
            if self._flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            
            # XXX: A hack here, I don't know why????
            im = im.transpose(Image.FLIP_TOP_BOTTOM)

            ImageClip(im, anchor=ImageAnchor.C).paste_onto(canvas, pt)
            
        # Debug
        if self._p1:
            p1c = wgs2px_relative(
                self._zoom + bin(self._msaa)[2:].count('0'), 
                self._p1, self._center,
                np.array(self.screen_size) * self._msaa,
            )
            p2c = wgs2px_relative(
                self._zoom + bin(self._msaa)[2:].count('0'), 
                self._p2, self._center,
                np.array(self.screen_size) * self._msaa,
            )
            cv2.circle(canvas, p1c, 10 * self._msaa, (1, 0, 0, 1), -1)
            cv2.circle(canvas, p2c, 10 * self._msaa, (0, 1, 0, 1), -1)

        if self._msaa > 1:
            canvas = cv2.resize(canvas, self.screen_size, interpolation=cv2.INTER_AREA)
        
        self._frame += 1
        self._frame_cache = canvas
        return canvas
    
        
class AbsoluteLayer(Layer):
    ''' An overlay layer with absolute screen coordinates. (i.e. don't move with canvas) '''
    
    # Don't respond to these.
    def set_center(self, center: Tuple[float, float]): pass
    def set_zoom(self, zoom: float): pass

class ImageAbsoluteLayer(AbsoluteLayer):
    def __init__(self, screen_size: Tuple[int, int], zoom: float):
        AbsoluteLayer.__init__(self, screen_size, zoom)
        self._images = []
    
    def add_image(self, coord: Tuple[int, int], arr: np.ndarray, zindex: int=-1):
        assert 0 <= coord[0] < self.screen_size[0]
        assert 0 <= coord[1] < self.screen_size[1]
        
        coord = np.asarray(coord)
        
        if zindex >= 0:
            self._images.insert(zindex, (coord, arr))
        else:
            self._images.append((coord, arr))
        self._frame_cache = None
        
    def del_image(self, arr: np.ndarray):
        for n, (c, a) in enumerate(self._images):
            if a is arr:
                del self._images[n]
                self._frame_cache = None
                break
        else:
            raise ValueError('Image does not exist.')
            
    def clear(self):
        self._images = []
        self._frame_cache = None
        
    def render_next_frame(self) -> np.ndarray:
        ''' Retrieve next frame '''
        if self._frame_cache is not None:
            return self._frame_cache
        
        canvas = np.zeros((self.screen_size[1], self.screen_size[0], 4), dtype='float32') # RGBA
        
        for c, a in self._images:
            # Last image on top.
            lt = c[::-1]
            rb = np.minimum(lt + a.shape[:2], canvas.shape[:2])
            a_rb = rb - lt + 0

            canvas[lt[0]:rb[0], lt[1]:rb[1], ...] = alpha_composite(
                canvas[lt[0]:rb[0], lt[1]:rb[1], ...],
                a[:a_rb[0], :a_rb[1], ...],
            )
        
        self._frame += 1
        self._frame_cache = canvas
        return canvas

class HUDLayer(ImageAbsoluteLayer):
    ''' HUD Layer to display fixed format information. Currently only Road sign is supported. '''
    def __init__(self, screen_size: Tuple[int, int], zoom: float):
        ImageAbsoluteLayer.__init__(self, screen_size, zoom)
        self._text = ''
        self._frame_cache_2 = None
    
    def set_road_icons(self, *road_icons: List[np.ndarray]):
        self.clear()
        
        x = int(0.025 * self.screen_size[0]) # 1920 => 48
        y = int(0.05 * self.screen_size[1]) # 1080 => 54
        
        xmargin = int(0.01 * self.screen_size[0]) # 1920 => 19
        
        for i in road_icons:
            self.add_image((x, y), i)
            x += i.shape[1] + xmargin
        
    def set_hud_text(self, text: str):
        self._text = text
        self._frame_cache = None
        self._frame_cache_2 = None
        
    def render_next_frame(self) -> np.ndarray:
        # handle frame cache carefully.
        # use a second layer cache.
        
        if self._frame_cache_2 is not None:
            return self._frame_cache_2
        
        # call base method first.
        canvas = ImageAbsoluteLayer.render_next_frame(self)
        
        # Revert frame count
        self._frame -= 1
        
        if self._text:
            x = int(0.900 * self.screen_size[0]) # 1920 => 1728
            y = int(0.050 * self.screen_size[1]) # 1080 => 54
            
            # Draw text
            canvas = draw_text(canvas, (x, y), self._text, size=1, width=2, color=(1, 0, 0, 1))
        
        self._frame += 1
        self._frame_cache2 = canvas
        return canvas
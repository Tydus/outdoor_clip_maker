from movie_maker import *
from util import *
from layers import *

from _Allv2_route import V, E

def test_bbcenterzoom(screen_size=(1920, 1080)):
    
    V = {
        r'Home': PoI((35.646004, 138.550669), r'Home', POI_ICONS['Lodging']),
        r'ASTY Gifu': PoI((35.409372, 136.757902), r'ASTY Gifu', POI_ICONS['Shopping mall']),
        r'Lake Biwa\nMarriott Hotel': PoI((35.124958, 135.954876), r'Lake Biwa\nMarriott Hotel', POI_ICONS['Hotel']),
        r'7-Eleven': PoI((34.932275, 135.790481), r'7-Eleven', POI_ICONS['Convenience store']),
        r'Fairfield\nOsaka Namba': PoI((34.662671, 135.496482), r'Fairfield\nOsaka Namba', POI_ICONS['Hotel']),
        r'Round 1': PoI((34.198399, 135.186066), r'Round 1', POI_ICONS['Video arcade']),
        r'Awaji SA': PoI((34.582674, 135.017499), r'Awaji SA', POI_ICONS['Rest stop']),
        r'Athena Kaigetsu': PoI((34.4394, 134.910893), r'Athena Kaigetsu', POI_ICONS['Hotel']),
        r'Naruto Park': PoI((34.236164, 134.64019), r'Naruto Park', POI_ICONS['Sightseeing spot']),
        r'Aeon Mall\nTokushima': PoI((34.061514, 134.573635), r'Aeon Mall\nTokushima', POI_ICONS['Shopping mall']),
        r'Round 1\nTakamatsu': PoI((34.338986, 134.067006), r'Round 1\nTakamatsu', POI_ICONS['Video arcade']),
        r'Kounoike SA': PoI((34.475268, 133.781379), r'Kounoike SA', POI_ICONS['Rest stop']),
        r'ANA Crowne Plaza Okayama, an IHG Hotel': PoI((34.666531, 133.916207), r'ANA Crowne Plaza Okayama, an IHG Hotel', POI_ICONS['Hotel']),
        r'Crowne Plaza\nOkayama': PoI((34.666531, 133.916207), r'Crowne Plaza\nOkayama', POI_ICONS['Hotel']),
        r'FamilyMart': PoI((34.62334, 133.871311), r'FamilyMart', POI_ICONS['Convenience store']),
        r'Sega Kurashiki': PoI((34.583882, 133.769983), r'Sega Kurashiki', POI_ICONS['Video arcade']),
        r'Round 1\nFukuyama': PoI((34.489177, 133.392503), r'Round 1\nFukuyama', POI_ICONS['Video arcade']),
        r'Fuji Grand\nOnomichi': PoI((34.425473, 133.238806), r'Fuji Grand\nOnomichi', POI_ICONS['Supermarket']),
        r'ENEOS\nMukaishima': PoI((34.395349, 133.196817), r'ENEOS\nMukaishima', POI_ICONS['Gas station']),
        r'Roadside station\nKazahaya-no-sato Fuwari': PoI((33.996906, 132.77232), r'Roadside station\nKazahaya-no-sato Fuwari', POI_ICONS['Rest stop']),
        r'Crowne Plaza\nMatsuyama': PoI((33.840872, 132.768936), r'Crowne Plaza\nMatsuyama', POI_ICONS['Hotel']),
        r'Dogo Onsen': PoI((33.850754, 132.785468), r'Dogo Onsen', POI_ICONS['Onsen']),
    }
    
    bb = Path.get_bounding_box(Path(*V.values()))
    print(bb)
    
    Movie(screen_size=(1920, 1080), fps=60, output_filename='test.mp4', center=V[r'Home'].location, zoom=15, pois=V, paths={}, animations=[
        (1, PanZoomTo(1, bounding_box=bb)), # single frame
    ], no_video=False)

def test_melayer():
    E2 = {
        'Matsuyama_Port': E['Matsuyama_Port'],
        'Ship': E['Ship'],
    }
        
    Movie(screen_size=(1920, 1080), fps=60, output_filename='test.mp4', center=V[r'Home'].location, zoom=12, pois={}, paths=E2, animations=[
        (' 00:00', MoveAlong('5', E['Matsuyama_Port'])), # 65 km
        (' 00:02', PanZoomTo('2', zoom=8)),
        (' 00:05', MoveAlong('5', E['Ship'])), # 88 km
        (' 00:08', PanZoomTo('2', zoom=15)),
    ])

def test_panzoom():
    Movie(screen_size=(1920, 1080), fps=5, output_filename='test.mp4', center=V[r'Fairfield\nOsaka Namba'].location, zoom=12, pois=V, paths=E, animations=[
        ('+00:00', MoveAlong('10', E['Yonago_Conan'])), # 45 km
        #('+00:04', PanZoomTo('01', zoom=15.000)),
    ],no_video=True)

def test_timeline():
    Movie(screen_size=(1920, 1080), fps=10, output_filename='test.mp4', center=V[r'Home'].location, zoom=12, pois=V, paths=E, animations=[
        (' 00:00', MoveAlong('10', E['Yonago_Conan'])), # 45 km
        ('+00:00', PanZoomTo('02', zoom=12.000)),
        ('-00:02', PanZoomTo('02', zoom=15.000)),
        
        ('|00:05', MoveAlong('10', E['Conan_Rokko'])), # 225 km
        ('+00:00', PanZoomTo('02', zoom=11.000)),
        ('-00:02', PanZoomTo('02', zoom=15.000)),
        
        ('|00:05', MoveAlong('10', E['Rokko_Namba'])), # 31 km
        ('+00:00', PanZoomTo('02', zoom=12.000)),
        ('-00:02', PanZoomTo('02', zoom=15.000)),
        
        ('|00:05', PanZoomTo('02', zoom=10.000)),
        ('+00:01', MoveAlong('10', E['Namba_Home'])), # 449 km
        ('+00:09', PanZoomTo('02', zoom=15.000)),
        
        ('|00:00', PanZoomTo('10', )), # sleep 10s
        
        ('|00:00', PanZoomTo('02', bounding_box=Path.get_bounding_box(*E.values()))), # Zoom to aLL edges.
        ('+00:00', CustomAni('00', prepare_cb=lambda movie: movie.canvas.find_first_layer_by_type(PathLayer).set_highlight(*E.values()))), # Highlight all edges.
        ('+00:00', CustomAni('00', prepare_cb=lambda movie: movie.canvas.find_first_layer_by_type(PoiLayer).set_name_visibility('none'))), # Hide all poi names, but keep icons.
        ('|00:00', PanZoomTo('05', )), # sleep 5s
        ('+00:00', CustomAni('00', prepare_cb=lambda movie: movie.canvas.find_first_layer_by_type(PoiLayer).clear())),
        
    ])
    
def test_image():
    Movie(screen_size=(1920, 1080), fps=10, output_filename='test.mp4', center=V[r'Home'].location, zoom=12, pois=V, paths=E, animations=[
        (' 00:00', MoveAlong('10', E['Yonago_Conan'])), # 45 km
        (' 00:03', ShowImage('5', ImageClip('assets/allv2/go/07-05-1.png', zoom=(1920, 1080)).array, (0, 0))),
    ])
    
if __name__ == '__main__':
    import sys
    globals()['test_%s' % sys.argv[1]]()
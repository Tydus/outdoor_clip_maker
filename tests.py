from movie_maker import *
from util import *

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
    
    bb = Path.get_bounding_box(Path(v.values()))
    
    Movie(screen_size=(1920, 1080), fps=60, center=V[r'Home'].location, zoom=15, pois=V, paths={}, animations=[
        (1, PanZoomTo(1, bounding_box=bb)), # single frame
    ], filename_prefix='test_')

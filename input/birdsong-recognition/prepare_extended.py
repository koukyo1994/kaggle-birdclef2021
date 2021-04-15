import warnings

import librosa
import pandas as pd
import soundfile as sf

from pathlib import Path
from joblib import delayed, Parallel


NAME2CODE = {
    'Empidonax alnorum_Alder Flycatcher': 'aldfly',
    'Recurvirostra americana_American Avocet': 'ameavo',
    'Botaurus lentiginosus_American Bittern': 'amebit',
    'Corvus brachyrhynchos_American Crow': 'amecro',
    'Spinus tristis_American Goldfinch': 'amegfi',
    'Falco sparverius_American Kestrel': 'amekes',
    'Anthus rubescens_American Pipit': 'amepip',
    'Setophaga ruticilla_American Redstart': 'amered',
    'Turdus migratorius_American Robin': 'amerob',
    'Mareca americana_American Wigeon': 'amewig',
    'Scolopax minor_American Woodcock': 'amewoo',
    'Spizelloides arborea_American Tree Sparrow': 'amtspa',
    "Calypte anna_Anna's Hummingbird": 'annhum',
    'Myiarchus cinerascens_Ash-throated Flycatcher': 'astfly',
    "Calidris bairdii_Baird's Sandpiper": 'baisan',
    'Haliaeetus leucocephalus_Bald Eagle': 'baleag',
    'Icterus galbula_Baltimore Oriole': 'balori',
    'Riparia riparia_Bank Swallow': 'banswa',
    'Hirundo rustica_Barn Swallow': 'barswa',
    'Mniotilta varia_Black-and-white Warbler': 'bawwar',
    'Megaceryle alcyon_Belted Kingfisher': 'belkin1',
    "Artemisiospiza belli_Bell's Sparrow": 'belspa2',
    "Thryomanes bewickii_Bewick's Wren": 'bewwre',
    'Coccyzus erythropthalmus_Black-billed Cuckoo': 'bkbcuc',
    'Pica hudsonia_Black-billed Magpie': 'bkbmag1',
    'Setophaga fusca_Blackburnian Warbler': 'bkbwar',
    'Poecile atricapillus_Black-capped Chickadee': 'bkcchi',
    'Archilochus alexandri_Black-chinned Hummingbird': 'bkchum',
    'Pheucticus melanocephalus_Black-headed Grosbeak': 'bkhgro',
    'Setophaga striata_Blackpoll Warbler': 'bkpwar',
    'Amphispiza bilineata_Black-throated Sparrow': 'bktspa',
    'Sayornis nigricans_Black Phoebe': 'blkpho',
    'Passerina caerulea_Blue Grosbeak': 'blugrb1',
    'Cyanocitta cristata_Blue Jay': 'blujay',
    'Molothrus ater_Brown-headed Cowbird': 'bnhcow',
    'Dolichonyx oryzivorus_Bobolink': 'boboli',
    "Chroicocephalus philadelphia_Bonaparte's Gull": 'bongul',
    'Strix varia_Barred Owl': 'brdowl',
    "Euphagus cyanocephalus_Brewer's Blackbird": 'brebla',
    "Spizella breweri_Brewer's Sparrow": 'brespa',
    'Certhia americana_Brown Creeper': 'brncre',
    'Toxostoma rufum_Brown Thrasher': 'brnthr',
    'Selasphorus platycercus_Broad-tailed Hummingbird':
    'brthum', 'Buteo platypterus_Broad-winged Hawk': 'brwhaw',
    'Setophaga caerulescens_Black-throated Blue Warbler': 'btbwar',
    'Setophaga virens_Black-throated Green Warbler': 'btnwar',
    'Setophaga nigrescens_Black-throated Gray Warbler': 'btywar',
    'Bucephala albeola_Bufflehead': 'buffle',
    'Polioptila caerulea_Blue-gray Gnatcatcher': 'buggna',
    'Vireo solitarius_Blue-headed Vireo': 'buhvir',
    "Icterus bullockii_Bullock's Oriole": 'bulori',
    'Psaltriparus minimus_Bushtit': 'bushti',
    'Spatula discors_Blue-winged Teal': 'buwtea',
    'Vermivora cyanoptera_Blue-winged Warbler': 'buwwar',
    'Campylorhynchus brunneicapillus_Cactus Wren': 'cacwre',
    'Larus californicus_California Gull': 'calgul',
    'Callipepla californica_California Quail': 'calqua',
    'Setophaga tigrina_Cape May Warbler': 'camwar',
    'Branta canadensis_Canada Goose': 'cangoo',
    'Cardellina canadensis_Canada Warbler': 'canwar',
    'Catherpes mexicanus_Canyon Wren': 'canwre',
    'Thryothorus ludovicianus_Carolina Wren': 'carwre',
    "Haemorhous cassinii_Cassin's Finch": 'casfin',
    'Hydroprogne caspia_Caspian Tern': 'caster1',
    "Vireo cassinii_Cassin's Vireo": 'casvir',
    'Bombycilla cedrorum_Cedar Waxwing': 'cedwax',
    'Spizella passerina_Chipping Sparrow': 'chispa',
    'Chaetura pelagica_Chimney Swift': 'chiswi',
    'Setophaga pensylvanica_Chestnut-sided Warbler': 'chswar',
    'Alectoris chukar_Chukar': 'chukar',
    "Nucifraga columbiana_Clark's Nutcracker": 'clanut',
    'Petrochelidon pyrrhonota_Cliff Swallow': 'cliswa',
    'Bucephala clangula_Common Goldeneye': 'comgol',
    'Quiscalus quiscula_Common Grackle': 'comgra',
    'Gavia immer_Common Loon': 'comloo',
    'Mergus merganser_Common Merganser': 'commer',
    'Chordeiles minor_Common Nighthawk': 'comnig',
    'Corvus corax_Common Raven': 'comrav',
    'Acanthis flammea_Common Redpoll': 'comred',
    'Sterna hirundo_Common Tern': 'comter',
    'Geothlypis trichas_Common Yellowthroat': 'comyel',
    "Accipiter cooperii_Cooper's Hawk": 'coohaw',
    "Calypte costae_Costa's Hummingbird": 'coshum',
    'Aphelocoma californica_California Scrub-Jay': 'cowscj1',
    'Junco hyemalis_Dark-eyed Junco': 'daejun',
    'Phalacrocorax auritus_Double-crested Cormorant': 'doccor',
    'Dryobates pubescens_Downy Woodpecker': 'dowwoo',
    'Empidonax oberholseri_Dusky Flycatcher': 'dusfly',
    'Podiceps nigricollis_Eared Grebe': 'eargre',
    'Sialia sialis_Eastern Bluebird': 'easblu',
    'Tyrannus tyrannus_Eastern Kingbird': 'easkin',
    'Sturnella magna_Eastern Meadowlark': 'easmea',
    'Sayornis phoebe_Eastern Phoebe': 'easpho',
    'Pipilo erythrophthalmus_Eastern Towhee': 'eastow',
    'Contopus virens_Eastern Wood-Pewee': 'eawpew',
    'Streptopelia decaocto_Eurasian Collared-Dove': 'eucdov',
    'Sturnus vulgaris_European Starling': 'eursta',
    'Coccothraustes vespertinus_Evening Grosbeak': 'evegro',
    'Spizella pusilla_Field Sparrow': 'fiespa',
    'Corvus ossifragus_Fish Crow': 'fiscro',
    'Passerella iliaca_Fox Sparrow': 'foxspa',
    'Mareca strepera_Gadwall': 'gadwal',
    'Leucosticte tephrocotis_Gray-crowned Rosy-Finch': 'gcrfin',
    'Pipilo chlorurus_Green-tailed Towhee': 'gnttow',
    'Anas crecca_Green-winged Teal': 'gnwtea',
    'Regulus satrapa_Golden-crowned Kinglet': 'gockin',
    'Zonotrichia atricapilla_Golden-crowned Sparrow': 'gocspa',
    'Aquila chrysaetos_Golden Eagle': 'goleag',
    'Ardea herodias_Great Blue Heron': 'grbher3',
    'Myiarchus crinitus_Great Crested Flycatcher': 'grcfly',
    'Ardea alba_Great Egret': 'greegr',
    'Geococcyx californianus_Greater Roadrunner': 'greroa',
    'Tringa melanoleuca_Greater Yellowlegs': 'greyel',
    'Bubo virginianus_Great Horned Owl': 'grhowl',
    'Butorides virescens_Green Heron': 'grnher',
    'Quiscalus mexicanus_Great-tailed Grackle': 'grtgra',
    'Dumetella carolinensis_Gray Catbird': 'grycat',
    'Empidonax wrightii_Gray Flycatcher': 'gryfly',
    'Dryobates villosus_Hairy Woodpecker': 'haiwoo',
    "Empidonax hammondii_Hammond's Flycatcher": 'hamfly',
    'Larus argentatus_Herring Gull': 'hergul',
    'Catharus guttatus_Hermit Thrush': 'herthr',
    'Lophodytes cucullatus_Hooded Merganser': 'hoomer',
    'Setophaga citrina_Hooded Warbler': 'hoowar',
    'Podiceps auritus_Horned Grebe': 'horgre',
    'Eremophila alpestris_Horned Lark': 'horlar',
    'Haemorhous mexicanus_House Finch': 'houfin',
    'Passer domesticus_House Sparrow': 'houspa',
    'Troglodytes aedon_House Wren': 'houwre',
    'Passerina cyanea_Indigo Bunting': 'indbun',
    'Baeolophus ridgwayi_Juniper Titmouse': 'juntit1',
    'Charadrius vociferus_Killdeer': 'killde',
    'Dryobates scalaris_Ladder-backed Woodpecker': 'labwoo',
    'Chondestes grammacus_Lark Sparrow': 'larspa',
    'Passerina amoena_Lazuli Bunting': 'lazbun',
    'Ixobrychus exilis_Least Bittern': 'leabit',
    'Empidonax minimus_Least Flycatcher': 'leafly',
    'Calidris minutilla_Least Sandpiper': 'leasan',
    "Toxostoma lecontei_LeConte's Thrasher": 'lecthr',
    'Spinus psaltria_Lesser Goldfinch': 'lesgol',
    'Chordeiles acutipennis_Lesser Nighthawk': 'lesnig',
    'Tringa flavipes_Lesser Yellowlegs': 'lesyel',
    "Melanerpes lewis_Lewis's Woodpecker": 'lewwoo',
    "Melospiza lincolnii_Lincoln's Sparrow": 'linspa',
    'Numenius americanus_Long-billed Curlew': 'lobcur',
    'Limnodromus scolopaceus_Long-billed Dowitcher': 'lobdow',
    'Lanius ludovicianus_Loggerhead Shrike': 'logshr',
    'Clangula hyemalis_Long-tailed Duck': 'lotduc',
    'Parkesia motacilla_Louisiana Waterthrush': 'louwat',
    "Geothlypis tolmiei_MacGillivray's Warbler": 'macwar',
    'Setophaga magnolia_Magnolia Warbler': 'magwar',
    'Anas platyrhynchos_Mallard': 'mallar3',
    'Cistothorus palustris_Marsh Wren': 'marwre',
    'Falco columbarius_Merlin': 'merlin',
    'Sialia currucoides_Mountain Bluebird': 'moublu',
    'Poecile gambeli_Mountain Chickadee': 'mouchi',
    'Zenaida macroura_Mourning Dove': 'moudov',
    'Cardinalis cardinalis_Northern Cardinal': 'norcar',
    'Colaptes auratus_Northern Flicker': 'norfli',
    'Circus hudsonius_Northern Harrier': 'norhar2',
    'Mimus polyglottos_Northern Mockingbird': 'normoc',
    'Setophaga americana_Northern Parula': 'norpar',
    'Anas acuta_Northern Pintail': 'norpin',
    'Spatula clypeata_Northern Shoveler': 'norsho',
    'Parkesia noveboracensis_Northern Waterthrush': 'norwat',
    'Stelgidopteryx serripennis_Northern Rough-winged Swallow': 'nrwswa',
    "Dryobates nuttallii_Nuttall's Woodpecker": 'nutwoo',
    'Contopus cooperi_Olive-sided Flycatcher': 'olsfly',
    'Leiothlypis celata_Orange-crowned Warbler': 'orcwar',
    'Pandion haliaetus_Osprey': 'osprey',
    'Seiurus aurocapilla_Ovenbird': 'ovenbi1',
    'Setophaga palmarum_Palm Warbler': 'palwar',
    'Empidonax difficilis_Pacific-slope Flycatcher': 'pasfly',
    'Calidris melanotos_Pectoral Sandpiper': 'pecsan',
    'Falco peregrinus_Peregrine Falcon': 'perfal',
    'Phainopepla nitens_Phainopepla': 'phaino',
    'Podilymbus podiceps_Pied-billed Grebe': 'pibgre',
    'Dryocopus pileatus_Pileated Woodpecker': 'pilwoo',
    'Pinicola enucleator_Pine Grosbeak': 'pingro',
    'Gymnorhinus cyanocephalus_Pinyon Jay': 'pinjay',
    'Spinus pinus_Pine Siskin': 'pinsis',
    'Setophaga pinus_Pine Warbler': 'pinwar',
    'Vireo plumbeus_Plumbeous Vireo': 'plsvir',
    'Setophaga discolor_Prairie Warbler': 'prawar',
    'Haemorhous purpureus_Purple Finch': 'purfin',
    'Sitta pygmaea_Pygmy Nuthatch': 'pygnut',
    'Mergus serrator_Red-breasted Merganser': 'rebmer',
    'Sitta canadensis_Red-breasted Nuthatch': 'rebnut',
    'Sphyrapicus ruber_Red-breasted Sapsucker': 'rebsap',
    'Melanerpes carolinus_Red-bellied Woodpecker': 'rebwoo',
    'Loxia curvirostra_Red Crossbill': 'redcro',
    'Aythya americana_Redhead': 'redhea',
    'Vireo olivaceus_Red-eyed Vireo': 'reevir1',
    'Phalaropus lobatus_Red-necked Phalarope': 'renpha',
    'Buteo lineatus_Red-shouldered Hawk': 'reshaw',
    'Buteo jamaicensis_Red-tailed Hawk': 'rethaw',
    'Agelaius phoeniceus_Red-winged Blackbird': 'rewbla',
    'Larus delawarensis_Ring-billed Gull': 'ribgul',
    'Aythya collaris_Ring-necked Duck': 'rinduc',
    'Pheucticus ludovicianus_Rose-breasted Grosbeak': 'robgro',
    'Columba livia_Rock Pigeon': 'rocpig',
    'Salpinctes obsoletus_Rock Wren': 'rocwre',
    'Archilochus colubris_Ruby-throated Hummingbird': 'rthhum',
    'Regulus calendula_Ruby-crowned Kinglet': 'ruckin',
    'Oxyura jamaicensis_Ruddy Duck': 'rudduc',
    'Bonasa umbellus_Ruffed Grouse': 'rufgro',
    'Selasphorus rufus_Rufous Hummingbird': 'rufhum',
    'Euphagus carolinus_Rusty Blackbird': 'rusbla',
    'Artemisiospiza nevadensis_Sagebrush Sparrow': 'sagspa1',
    'Oreoscoptes montanus_Sage Thrasher': 'sagthr',
    'Passerculus sandwichensis_Savannah Sparrow': 'savspa',
    "Sayornis saya_Say's Phoebe": 'saypho',
    'Piranga olivacea_Scarlet Tanager': 'scatan',
    "Icterus parisorum_Scott's Oriole": 'scoori',
    'Charadrius semipalmatus_Semipalmated Plover': 'semplo',
    'Calidris pusilla_Semipalmated Sandpiper': 'semsan',
    'Asio flammeus_Short-eared Owl': 'sheowl',
    'Accipiter striatus_Sharp-shinned Hawk': 'shshaw',
    'Plectrophenax nivalis_Snow Bunting': 'snobun',
    'Anser caerulescens_Snow Goose': 'snogoo',
    'Tringa solitaria_Solitary Sandpiper': 'solsan',
    'Melospiza melodia_Song Sparrow': 'sonspa',
    'Porzana carolina_Sora': 'sora',
    'Actitis macularius_Spotted Sandpiper': 'sposan',
    'Pipilo maculatus_Spotted Towhee': 'spotow',
    "Cyanocitta stelleri_Steller's Jay": 'stejay',
    "Buteo swainsoni_Swainson's Hawk": 'swahaw',
    'Melospiza georgiana_Swamp Sparrow': 'swaspa',
    "Catharus ustulatus_Swainson's Thrush": 'swathr',
    'Tachycineta bicolor_Tree Swallow': 'treswa',
    'Cygnus buccinator_Trumpeter Swan': 'truswa',
    'Baeolophus bicolor_Tufted Titmouse': 'tuftit',
    'Cygnus columbianus_Tundra Swan': 'tunswa',
    'Catharus fuscescens_Veery': 'veery',
    'Pooecetes gramineus_Vesper Sparrow': 'vesspa',
    'Tachycineta thalassina_Violet-green Swallow': 'vigswa',
    'Vireo gilvus_Warbling Vireo': 'warvir',
    'Sialia mexicana_Western Bluebird': 'wesblu',
    'Aechmophorus occidentalis_Western Grebe': 'wesgre',
    'Tyrannus verticalis_Western Kingbird': 'weskin',
    'Sturnella neglecta_Western Meadowlark': 'wesmea',
    'Calidris mauri_Western Sandpiper': 'wessan',
    'Piranga ludoviciana_Western Tanager': 'westan',
    'Contopus sordidulus_Western Wood-Pewee': 'wewpew',
    'Sitta carolinensis_White-breasted Nuthatch': 'whbnut',
    'Zonotrichia leucophrys_White-crowned Sparrow': 'whcspa',
    'Plegadis chihi_White-faced Ibis': 'whfibi',
    'Zonotrichia albicollis_White-throated Sparrow': 'whtspa',
    'Aeronautes saxatalis_White-throated Swift': 'whtswi',
    'Empidonax traillii_Willow Flycatcher': 'wilfly',
    "Gallinago delicata_Wilson's Snipe": 'wilsni1',
    'Meleagris gallopavo_Wild Turkey': 'wiltur',
    'Troglodytes hiemalis_Winter Wren': 'winwre3',
    "Cardellina pusilla_Wilson's Warbler": 'wlswar',
    'Aix sponsa_Wood Duck': 'wooduc',
    "Aphelocoma woodhouseii_Woodhouse's Scrub-Jay": 'wooscj2',
    'Hylocichla mustelina_Wood Thrush': 'woothr',
    'Fulica americana_American Coot': 'y00475',
    'Empidonax flaviventris_Yellow-bellied Flycatcher': 'yebfly',
    'Sphyrapicus varius_Yellow-bellied Sapsucker': 'yebsap',
    'Xanthocephalus xanthocephalus_Yellow-headed Blackbird': 'yehbla',
    'Setophaga petechia_Yellow Warbler': 'yelwar',
    'Setophaga coronata_Yellow-rumped Warbler': 'yerwar',
    'Vireo flavifrons_Yellow-throated Vireo': 'yetvir'
}


def resample(df: pd.DataFrame, audio_dir: Path, target_sr=32000):
    resample_dir = Path("train_audio_resampled")
    resample_dir.mkdir(exist_ok=True, parents=True)
    warnings.simplefilter("ignore")

    for i, row in df.iterrows():
        ebird_code = row.ebird_code
        filename = row.filename
        ebird_dir = resample_dir / ebird_code
        if not ebird_dir.exists():
            ebird_dir.mkdir(exist_ok=True, parents=True)

        try:
            y, _ = librosa.load(
                audio_dir / ebird_code / filename,
                sr=target_sr,
                mono=True,
                res_type="kaiser_fast")

            filename = filename.replace(".mp3", ".wav")
            sf.write(ebird_dir / filename, y, samplerate=target_sr)
        except Exception:
            with open("skipped.txt", "a") as f:
                file_path = str(audio_dir / ebird_code / filename)
                f.write(file_path + "\n")


def replace_secondary_labels(s: str):
    s_ = eval(s)
    new_s = []
    for ss in s_:
        if NAME2CODE.get(ss) is not None:
            new_s.append(NAME2CODE[ss])
    return str(new_s)


if __name__ == "__main__":
    train = pd.read_csv("train_extended.csv")
    a_m_dir = Path("A-M")
    n_z_dir = Path("N-Z")

    a_m = train.query("ebird_code < 'n'").reset_index(drop=True)
    n_z = train.query("ebird_code >= 'n'").reset_index(drop=True)
    for trn, audio_dir in zip([a_m, n_z], [a_m_dir, n_z_dir]):
        dfs = []
        njobs = 20
        for i in range(njobs):
            if i == njobs-1:
                start = i * (len(trn) // njobs)
                df = trn.iloc[start:, :].reset_index(drop=True)
                dfs.append(df)
            else:
                start = i * (len(trn) // njobs)
                end = (i + 1) * (len(trn) // njobs)
                df = trn.iloc[start:end, :].reset_index(drop=True)
                dfs.append(df)

        Parallel(n_jobs=njobs, verbose=10)(
            delayed(resample)(df, audio_dir, 32000) for df in dfs)

    new_train = train[
        ["ebird_code", "secondary_labels", "type", "latitude", "longitude",
         "sci_name", "species", "author", "date", "filename", "license",
         "rating", "time", "url"]]
    new_train.columns = [
        "primary_label", "secondary_labels", "type", "latitude", "longitude",
        "scientific_name", "common_name", "author", "date", "filename", "license",
        "rating", "time", "url"
    ]

    new_train["filename"] = new_train["filename"].map(
        lambda x: x.replace(".mp3", ".wav"))
    new_train["secondary_labels"] = new_train["secondary_labels"].map(
        replace_secondary_labels)
    resampled = pd.read_csv("train_resampled.csv")
    new_train = pd.concat([new_train, resampled],
                          axis=0).reset_index(drop=True)
    new_train.to_csv("train_resampled_extended.csv", index=False)

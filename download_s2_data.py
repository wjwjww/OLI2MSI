import numpy as np
from tqdm import tqdm
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt, InvalidChecksumError
from datetime import date


USERNAME = ''
PASSWORD = ''
api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')
geojson_path = './geojson_files/moutain.geojson'    # the .geojson file can be created from http://geojson.io
start_date = '20190701'
end_date = '20190901'
save_dir = 'D:/Sentinel2_China'

# search by polygon, time, and SciHub query keywords
footprint = geojson_to_wkt(read_geojson(geojson_path))
products = api.query(area=footprint,
                     date=(start_date, end_date),
                     platformname='Sentinel-2',
                     producttype='S2MSI1C',
                     cloudcoverpercentage=(0, 3),
                     )

# -----------------------------------
api.download(list(products)[0])
# -----------------------------------
# convert to Pandas DataFrame
valid_products = []
products_tiles = api.to_dataframe(products)['tileid']
for tileid in tqdm(set(products_tiles)):
    if isinstance(tileid, str):
        tem_ls = list(products_tiles[products_tiles == tileid].index)
        if len(tem_ls) == 1:
            valid_products += tem_ls
        else:
            tem_size = []
            for tem_id in tem_ls:
                tem_size.append(api.get_product_odata(tem_id)['size'])
            max_id = tem_ls[np.array(tem_size).argmax()]
            valid_products.append(max_id)
    else:
        continue

print(f'valid products: {len(valid_products)} of {len(products)}')
for i, product in enumerate(valid_products):
    print(f'downloading...: {i}/{len(valid_products)}')
    # download single scene by known product id
    try:
        api.download(product, directory_path=save_dir)
    except InvalidChecksumError:
        continue

# # GeoJSON FeatureCollection containing footprints and metadata of the scenes
# products_js = api.to_geojson(products)
#
# # GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
products_geo = api.to_geodataframe(products)

# # Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# # its download url
# odata_s = api.get_product_odata(list(products)[0])

# # Get the product's full metadata available on the server
# odata_l = api.get_product_odata(list(products)[0], full=True)

import os
import datetime

import shapely.wkt
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, mapping
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt, InvalidChecksumError

import os
import shutil
import tarfile
from homura import download
import requests
from bs4 import BeautifulSoup


# LANDSAT_METADATA_URL = 'http://storage.googleapis.com/gcp-public-data-landsat/index.csv.gz'
# SENTINEL2_METADATA_URL = 'http://storage.googleapis.com/gcp-public-data-sentinel-2/index.csv.gz'
# AWS LANDSAT-8 SCENE LIST URL: 'http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz'
start_date = '20190920'
end_date = '20190925'
crs = 'EPSG:4326'
max_cloud_cover = 10.
paths = [126, 126, 126, 126, 126, 127, 127, 127, 128, 128]
rows = [38, 39, 40, 41, 42, 40, 41, 42, 41, 42]
# paths = [120]
# rows = [39]
save_l8_dir = r''
save_s2_dir = r'D:\pair_dataset\sentinel2'


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class SentinelQuery(object):
    def __init__(self):
        self.api = SentinelAPI('wjw', 'w3.141592658', 'https://scihub.copernicus.eu/dhus')

    def __call__(self, geom, start_time, end_time):
        footprint = geom.to_wkt()
        products = self.api.query(area=footprint,
                                  date=(start_time, end_time),
                                  platformname='Sentinel-2',
                                  producttype='S2MSI1C'
                                  )

        df = self.api.to_dataframe(products)
        if len(products) == 0:
            return df
        geometry = [shapely.wkt.loads(fp) for fp in df['footprint']]
        # remove useless columns
        df.drop(['footprint', 'gmlfootprint'], axis=1, inplace=True)
        return gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    def download(self, s2_uuid, directory_path):
        product_info = self.api.download(s2_uuid, directory_path)
        if not self.api._md5_compare(product_info['path'], product_info['md5']):
            # os.remove(product_info['path'])
            print('File corrupt: checksums do not match, remove and re-download')
            # self.download(s2_uuid, directory_path)


def AWS_download(aws_scene, save_dir='.'):

    # Print some the product ID
    print('\n', 'EntityId:', aws_scene.productId, '\n')
    print(' Checking content: ', '\n')

    # Request the html text of the download_url from the amazon server.
    # download_url example: https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/index.html
    # download_url = f'https://landsat-pds.s3.amazonaws.com/c1/L8/{row.entityId[3:6]}/{row.entityId[6:9]}/{row.productId}/index.html'
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
    #     'Host': 'landsat-pds.s3.amazonaws.com'
    # }

    download_url = aws_scene.download_url
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
    #     'Host': 's3-us-west-2.amazonaws.com'
    # }
    response = requests.get(download_url)

    # If the response status code is fine (200)
    if response.status_code == 200:

        # Import the html to beautiful soup
        html = BeautifulSoup(response.content, 'html.parser')

        # Create the dir where we will put this image files.
        entity_dir = os.path.join(save_dir, aws_scene.productId)
        os.makedirs(entity_dir, exist_ok=True)

        # Second loop: for each band of this image that we find using the html <li> tag
        file_list = []
        for li in html.find_all('li'):

            # Get the href tag
            file = li.find_next('a').get('href')
            if file.endswith('.IMD') or file.endswith('.ovr'):
                continue
            else:
                file_list.append(file)

        for f in file_list:
            # Download the files
            # code from: https://stackoverflow.com/a/18043472/5361345
            url = aws_scene.download_url.replace('index.html', f)
            status = requests.head(url).status_code
            if status != 200:
                raise ValueError
            download(url, os.path.join(entity_dir, f))

        source_dir = entity_dir
        output_filename = '{}.tar.gz'.format(aws_scene.productId)
        out_location = os.path.join(save_dir, output_filename)
        with tarfile.open(out_location, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        shutil.rmtree(source_dir)
        print('Successfully download: {}'.format(aws_scene.productId))


def main():
    all_l8_meta = pd.read_csv('./aws_scene_list.gz', compression='gzip')   # AWS LANDSAT-8 SCENE LIST URL: 'http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz'
    all_l8_meta = all_l8_meta[all_l8_meta.processingLevel == 'L1TP']
    query = SentinelQuery()
    for path, row in zip(paths, rows):
        l8_meta = all_l8_meta[(all_l8_meta.path == path) &
                              (all_l8_meta.row == row) &
                              (all_l8_meta.productId.str.endswith('_T1')) &
                              (all_l8_meta.cloudCover <= max_cloud_cover) &
                              (all_l8_meta.acquisitionDate > datetime.datetime.strptime(start_date, '%Y%m%d').strftime(
                                  '%Y-%m-%d %H:%M:%S.%f')) &
                              (all_l8_meta.acquisitionDate < datetime.datetime.strptime(end_date, '%Y%m%d').strftime(
                                  '%Y-%m-%d %H:%M:%S.%f'))
                              ]
        l8_meta = l8_meta.drop_duplicates()
        geometry = [
            box(*bbox)
            for bbox in zip(
                l8_meta.min_lon, l8_meta.min_lat, l8_meta.max_lon, l8_meta.max_lat
            )
        ]
        l8_catalog = gpd.GeoDataFrame(l8_meta, crs=crs, geometry=geometry)

        for ind, item in l8_catalog.iterrows():
            tem_dict = dict()
            total_size = 0.
            time_delta = datetime.timedelta(seconds=3600)
            sensing_time = datetime.datetime.strptime(item.acquisitionDate, '%Y-%m-%d %H:%M:%S.%f')
            s2_catalog = query(item.geometry, sensing_time - time_delta, sensing_time + time_delta)
            if len(s2_catalog) > 0:
                AWS_download(item, save_dir=save_l8_dir)
            for jnd, s2_item in s2_catalog.iterrows():
                tem_dict[s2_item['title']] = s2_item['size']
                total_size += float(s2_item['size'][:-3])
                query.download(s2_item.uuid, directory_path=save_s2_dir)
            num = len(s2_catalog)
            output = '[*]'
            output += item.productId
            output += ' Cloud:{:<6.2f}'.format(item.cloudCover)
            output += ' match {} sentinel2 tiles. total size: {:<6.2f} MB \n'.format(num, total_size)
            output += dict2str(tem_dict)
            print(output)


if __name__ == "__main__":
    main()

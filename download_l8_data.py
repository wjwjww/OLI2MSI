from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import os
import shutil
import tarfile
from tqdm import tqdm
from contextlib import closing
from homura import download
import requests
from bs4 import BeautifulSoup
from subprocess import Popen, PIPE


def visualize_intersects_region(footprint, intersections):
    xy = np.asarray(footprint.centroid[0].xy).squeeze()
    center = list(xy[::-1])

    # Create a map to visualize the paths and rows that intersects.
    # Select a zoom
    zoom = 6
    # Create the most basic OSM folium map
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)

    # Add the bounds GeoDataFrame in red
    m.add_child(folium.GeoJson(bounds.__geo_interface__, name='Area of Study',
                               style_function=lambda x: {'color': 'red', 'alpha': 0}))

    # Iterate through each Polygon of paths and rows intersecting the area
    for i, row in intersections.iterrows():
        # Create a string for the name containing the path and row of this Polygon
        name = 'path: %03d, row: %03d' % (row.PATH, row.ROW)
        # Create the folium geometry of this Polygon
        g = folium.GeoJson(row.geometry.__geo_interface__, name=name)
        # Add a folium Popup object with the name string
        g.add_child(folium.Popup(name))
        # Add the object to the map
        g.add_to(m)

    folium.LayerControl().add_to(m)
    m.save('./wrs.html')


def AWS_download(products, save_dir='.'):
    # For each row
    for i, row in products.iterrows():

        # Print some the product ID
        print('\n', 'EntityId:', row.productId, '\n')
        print(' Checking content: ', '\n')

        # Request the html text of the download_url from the amazon server.
        # download_url example: https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/index.html
        # download_url = f'https://landsat-pds.s3.amazonaws.com/c1/L8/{row.entityId[3:6]}/{row.entityId[6:9]}/{row.productId}/index.html'
        # headers = {
        #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
        #     'Host': 'landsat-pds.s3.amazonaws.com'
        # }

        download_url = row.download_url
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
            entity_dir = os.path.join(save_dir, row.productId)
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
                url = row.download_url.replace('index.html', f)
                status = requests.head(url).status_code
                if status != 200:
                    raise ValueError
                download(url, os.path.join(entity_dir, f))

        source_dir = entity_dir
        output_filename = '{}.tar.gz'.format(row.productId)
        out_location = os.path.join(save_dir, output_filename)
        with tarfile.open(out_location, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        shutil.rmtree(source_dir)
        print('Successfully download: {}'.format(row.productId))


# TODO: incomplete, complete it when needed
def GCS_download(products, save_dir='.'):
    # For each row
    for i, row in products.iterrows():

        # Print some the product ID
        print('\n', 'EntityId:', row.productId, '\n')
        print(' Checking content: ', '\n')

        # Request the html text of the download_url from the amazon server.
        # example url: 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B2.TIF'
        download_url = f'http://storage.googleapis.com/gcp-public-data-landsat/LC08/01/{row.SCENE_ID[3:6]}/{row.SCENE_ID[6:9]}/{row.PRODUCT_ID}'


# TODO: incomplete, complete it when needed
def GCS_download_with_gsutil(products, save_dir='.'):
    # example url: gs://gcp-public-data-landsat/LC08/01/044/034/LC08_L1GT_044034_20130330_20170310_01_T2/
    url = ''
    p = Popen('gsutil -m cp -r '+url+' ./data', stdout=PIPE, stderr=PIPE, shell=True)
    stdout = p.stdout.read()
    stderr = p.stderr.read()
    if stderr:
        raise NotImplementedError(stderr)
    if stdout:
        print(stdout)


def GCS_filter(wrs_intersection, start_date, end_date, cloud_cover=20, ):
    paths, rows = wrs_intersection['PATH'].values, wrs_intersection['ROW'].values
    # google_scenes, 'https://storage.googleapis.com/gcp-public-data-landsat/index.csv.gz'
    scenes = pd.read_csv('./index.csv.gz', compression='gzip')

    scenes.dropna(subset=['PRODUCT_ID'], inplace=True, axis=0)
    scenes = scenes[
        (scenes.SPACECRAFT_ID == 'LANDSAT_8') &
        (scenes.PRODUCT_ID.str.endswith('_T1')) &
        (scenes.DATA_TYPE == 'L1TP') &
        (scenes.COLLECTION_NUMBER == '01')
        ]
    # Empty list to add the images
    bulk_list = []

    # Iterate through paths and rows
    for path, row in zip(paths, rows):

        print('Path:', path, 'Row:', row)

        # Filter the Google cloud storage table for images matching path, row, cloud cover and processing state.
        tem_scenes = scenes[
            (scenes.WRS_PATH == path) &
            (scenes.WRS_ROW == row) &
            (scenes.CLOUD_COVER <= cloud_cover) &
            (scenes.DATE_ACQUIRED > datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')) &
            (scenes.DATE_ACQUIRED < datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d'))
            ]

        print(' Found {} images\n'.format(len(tem_scenes)))

        # If any scenes exists, select the one that have the minimum cloudCover.
        if len(tem_scenes):
            scene = tem_scenes.sort_values('CLOUD_COVER').iloc[0]
            # Add the selected scene to the bulk download list.
            bulk_list.append(scene)

    bulk_frame = pd.concat(bulk_list, 1).T
    return bulk_frame


def AWS_filter(wrs_intersection, start_date, end_date, cloud_cover=20):
    paths, rows = wrs_intersection['PATH'].values, wrs_intersection['ROW'].values
    # wrs_intersection_night = wrs_night[wrs_night.intersects(bounds.geometry[0])]
    # paths_night, rows_night = wrs_intersection_night['PATH'].values, wrs_intersection_night['ROW'].values

    # Checking Available Images on Amazon S3
    # 'http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz'
    s3_scenes = pd.read_csv('./aws_scene_list.gz', compression='gzip')
    s3_scenes = s3_scenes[
                       s3_scenes.productId.str.endswith('_T1')
                       ]
    # Empty list to add the images
    bulk_list = []

    # Iterate through paths and rows
    for path, row in zip(paths, rows):

        print('Path:', path, 'Row:', row)

        # Filter the Landsat Amazon S3 table for images matching path, row, cloud cover and processing state.
        scenes = s3_scenes[(s3_scenes.path == path) & (s3_scenes.row == row) &
                           (s3_scenes.cloudCover <= cloud_cover) &
                           (s3_scenes.productId.str.endswith('_T1')) &
                           (s3_scenes.acquisitionDate > datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d %H:%M:%S.%f')) &
                           (s3_scenes.acquisitionDate < datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d %H:%M:%S.%f'))
                           ]

        print(' Found {} images\n'.format(len(scenes)))

        # If any scenes exists, select the one that have the minimum cloudCover.
        if len(scenes):
            scene = scenes.sort_values('cloudCover').iloc[0]
            # Add the selected scene to the bulk download list.
            bulk_list.append(scene)

    bulk_frame = pd.concat(bulk_list, 1).T
    return bulk_frame


if __name__ == '__main__':
    # the .geojson file can be created from http://geojson.io
    bounds = gpd.read_file('./geojson_files/moutain.geojson')
    start_date = '20190701'
    end_date = '20190901'
    save_dir = r'./'
    # download from: https://www.usgs.gov/land-resources/nli/landsat/landsat-shapefiles-and-kml-files
    wrs_day = gpd.GeoDataFrame.from_file('./WRS2_descending/WRS2_descending.shp')
    # wrs_night = gpd.GeoDataFrame.from_file('./WRS2_ascending_nighttime_data/WRS2_acsending.shp')

    wrs_intersection = wrs_day[wrs_day.intersects(bounds.geometry[0])]

    # visualize the ROI
    # visualize_intersects_region(bounds, wrs_intersection)

    # # AWS download
    bulk_list = AWS_filter(wrs_intersection, start_date, end_date)
    AWS_download(bulk_list, save_dir)
    # ====================================================
    # GCS download
    # bulk_list = GCS_filter(wrs_intersection, start_date, end_date)

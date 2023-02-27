from dataclasses import dataclass
import geojson
from osgeo import gdal,osr
import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import glob
from typing import Union
from data_preparation import fetchData
import math

@dataclass(frozen=True, init=True)
class Node:
    lat:float
    lon:float
    
'''CycleWay is a dedicated bicycle road, that does not have to be adjusted'''
@dataclass(frozen=True, init=True)
class CycleWay:
    nodes:list
    MASK_VALUE = 255

'''Street is a regular street that has cycle lanes or tracks on the side'''
@dataclass(frozen=True, init=True)
class Street:
    nodes:list
    n_car_lanes:int
    bike_right:bool
    bike_left:bool
    MASK_VALUE = 255

class Dataprep:
    cycleways=[]
    streets=[]

    def addstreet(self, feature:any, nodes:list, lanes:int):
        lanes = int(round(float(lanes)))
        bike_right= 'cycleway:right' in feature['properties']
        bike_left= 'cycleway:left' in feature['properties'] 

        if 'cycleway:both' in feature['properties'] or ('sidewalk' in feature['properties'] and feature['properties']['sidewalk']=="both"):
            bike_right= True
            bike_left= True

        self.streets.append(Street(nodes, lanes, bike_right, bike_left))

def fetchData(path) -> list: 
    dataprep=Dataprep()

    #read export file as geojson
    with open(path, encoding='utf-8') as f:
        gf = geojson.load(f)

    #iterate over features and add to corresponding class
    for feature in gf['features']:

        #add all nodes of feature
        nodes = []
        for coordinate in feature['geometry']['coordinates']:

            #PYTHON AUTOMATICALLY ROUNDES THE COORDINATES
            lon = coordinate[0]
            lat = coordinate[1]
            nodes.append(Node(lat, lon))

        #decision if street or cycleway
        if 'lanes' in feature['properties']:
            lanes=feature['properties']['lanes']
            dataprep.addstreet(feature, nodes, lanes)

        elif 'lane_markings' in feature['properties']:
            lanes=2
            dataprep.addstreet(feature, nodes, lanes)

        else:
            dataprep.cycleways.append(CycleWay(nodes))
    
    ret = dataprep.cycleways
    ret.extend(dataprep.streets)
    return ret

DISTANCE_FROM_CENTER_LINE_PER_CAR_LANE = 17.625
MASK_LINE_WIDTH = 11.8125

OVERLAY_MASK = False

IMG_WIDTH = IMG_HEIGHT = 10_000

class GeoTif:
    width:int
    height:int
    name:str
    _coordTransform:any 
    _geo_transform_inv:any

    def print(self):
        print(self.name)
        print(self._coordTransform)
        print("inverse:")
        print(self._geo_transform_inv)

    def __init__(self, path:str):
        self.name = os.path.basename(path)

        src = gdal.Open(path)

        self.width = src.RasterXSize
        self.height = src.RasterYSize

        point_srs = osr.SpatialReference()
        point_srs.ImportFromEPSG(4326) # hardcode for lon/lat
        point_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)    

        file_srs = osr.SpatialReference()
        file_srs.ImportFromWkt(src.GetProjection())

        self._coordTransform = osr.CoordinateTransformation(point_srs, file_srs)
        self._geo_transform_inv = gdal.InvGeoTransform(src.GetGeoTransform())

        del src

    def toPixelCoord(self, lat:float, lon:float) -> tuple[int, int]:
        '''Takes latitude and longitude and returns corresponding (x,y) as pixel in the image'''
        point_x = lon
        point_y = lat
        mapx, mapy, _ = self._coordTransform.TransformPoint(point_x, point_y)

        pixel_x, pixel_y = gdal.ApplyGeoTransform(self._geo_transform_inv, mapx, mapy)

        # round to pixel
        pixel_x = round(pixel_x)
        pixel_y = round(pixel_y)

        return pixel_x, pixel_y


def extractLanes(street: Street, v_list:list) -> tuple[list, list]:
    left_track = []
    right_track = []
    
    for i in range(len(v_list) - 1):
        start = v_list[i]
        start = np.array(start)
        end = v_list[i+1]
        end = np.array(end)

        v = end - start
        v_x, v_y = v
        orthogonal_left = np.array([v_y, -v_x]) # vector turned by 90 deg clockwise in "upside-down" 2D
        div = math.sqrt(np.sum(orthogonal_left**2))
        orthogonal_left = orthogonal_left / div if div else np.array((0.,0.)) # make unit vector
        orthogonal_left *= street.n_car_lanes / 2 * DISTANCE_FROM_CENTER_LINE_PER_CAR_LANE + MASK_LINE_WIDTH / 2 # extend to desired length

        orthogonal_right = -orthogonal_left # vector turned by 90 deg counter-clockwise in "upside-down" 2D and extended to deisred length

        if street.bike_left:
            left_track.extend([tuple(start + orthogonal_left), tuple(end + orthogonal_left)])
        if street.bike_right:
            right_track.extend([tuple(start + orthogonal_right), tuple(end + orthogonal_right)])

    return left_track, right_track

def draw(canvas:ImageDraw.ImageDraw, road:Union[CycleWay,Street], vectors:list[tuple[int, int]]) -> None:
    width = round(MASK_LINE_WIDTH)
    if OVERLAY_MASK:
        fill = (255,0,0,100)
    else:
        fill = 255
    if type(road) is CycleWay:
        canvas.line(vectors, fill=fill, width=width)
    if type(road) is Street:
        left, right = extractLanes(road, vectors)
        canvas.line(left, fill=fill, width=width)
        canvas.line(right, fill=fill, width=width)


def createImage(path: str) -> Image:
    if not OVERLAY_MASK:
        return Image.new('L', (IMG_WIDTH, IMG_HEIGHT))
    else:
        return Image.open(path)


if __name__ == "__main__":
    path = "path/to/save/files/to"
    geotif = 'bikesat.geojson'

    path_to_tifs = os.path.join(path, "*.tif")
    tifs = [GeoTif(fname) for fname in glob.glob(path_to_tifs)]

    print("Found", len(tifs), "GeoTifs.")

    osm_data = fetchData(geotif)

    print("Loaded", len(osm_data), "individual road segments with, in total,", sum(len(i.nodes) for i in osm_data),"data points.")

    print("Creating masks by applying road segments...")
    for tif in tqdm(tifs, total=len(tifs)):
        mask = createImage(os.path.join(path, tif.name)) 
        canvas = ImageDraw.Draw(mask, mode="RGBA" if OVERLAY_MASK else mask.mode)
        
        for road in osm_data:
            vectors = [tif.toPixelCoord(node.lat, node.lon) for node in road.nodes] 
            draw(canvas, road, vectors)

        mask.save(os.path.join(path, tif.name[:-3] + "png"))
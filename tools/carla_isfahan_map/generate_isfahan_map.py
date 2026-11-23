#!/usr/bin/env python3
"""
سازنده نقشه تقاطع چهارباغ-آزادی اصفهان برای CARLA
تولید فایل OpenDRIVE (.xodr) دقیق با γεومتری واقعی، فازهای سیگنال، و پیاده‌روها
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
import json


@dataclass
class Vec2D:
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class LaneSpec:
    id: int
    type: str  # driving, bidirectional, sidewalk, border, restricted, parking, median, shoulder
    level: bool = False
    width: float = 3.5
    speed: float = 50.0  # km/h
    access_restriction: str = "none"  # none, bus, taxi, hov, emergency


class IsfahanIntersectionBuilder:
    """
    سازنده تقاطع چهارباغ-آزادی بر اساس مشخصات مهندسی شهر اصفهان
    مختصات: 32.6546° N, 51.6680° E
    """
    
    # أبعاد تقاطع (متر)
    INTERSECTION_WIDTH = 30.0      # عرض تقاطع (شامل پیاده‌روها)
    APPROACH_LENGTH = 200.0        # طول روی conoce
    LANE_WIDTH = 3.5               # عرض خط استاندارد
    BRT_LANE_WIDTH = 3.75          # عرض خط BRT (کمتر rộngتر)
    SIDEWALK_WIDTH = 4.5           # عرض پیاده‌رو
    MEDIAN_WIDTH = 2.0             # عرض میانه
    CROSSWALK_WIDTH = 4.5          # عرض خط پیاده‌رو
    
    # مختصات مرکز تقاطع
    CENTER = Vec2D(0.0, 0.0)
    
    def __init__(self, output_dir: str = "data/carla_maps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # تعریف مسیرها (Roads)
        self.roads = self._define_roads()
        self.junctions = self._define_junctions()
        self.controllers = self._define_controllers()
    
    def _define_roads(self) -> List[Dict]:
        """تعریف ۴ روی conoce اصلی + میانه‌ها"""
        roads = []
        
        # چهارباغ شمالی (شمال به جنوب) - ID: 100
        roads.append({
            'id': 100,
            'name': 'ChaharBagh_North',
            'junction': 1,
            'length': self.APPROACH_LENGTH + self.INTERSECTION_WIDTH,
            'lanes': self._create_chaharbag_lanes('northbound'),
            'geometry': self._create_road_geometry('north'),
            'traffic_lights': ['tl_100_nb_straight', 'tl_100_nb_left'],
        })
        
        # چهارباغ جنوبی (جنوب به شمال) - ID: 200
        roads.append({
            'id': 200,
            'name': 'ChaharBagh_South',
            'junction': 1,
            'length': self.APPROACH_LENGTH + self.INTERSECTION_WIDTH,
            'lanes': self._create_chaharbag_lanes('southbound'),
            'geometry': self._create_road_geometry('south'),
            'traffic_lights': ['tl_200_sb_straight', 'tl_200_sb_left'],
        })
        
        # آزادی شرقی (شرق به غرب) - ID: 300
        roads.append({
            'id': 300,
            'name': 'Azadi_East',
            'junction': 1,
            'length': self.APPROACH_LENGTH + self.INTERSECTION_WIDTH,
            'lanes': self._create_azadi_lanes('eastbound'),
            'geometry': self._create_road_geometry('east'),
            'traffic_lights': ['tl_300_eb_straight', 'tl_300_eb_left'],
        })
        
        # آزادی غربی (غرب به شرق) - ID: 400
        roads.append({
            'id': 400,
            'name': 'Azadi_West',
            'junction': 1,
            'length': self.APPROACH_LENGTH + self.INTERSECTION_WIDTH,
            'lanes': self._create_azadi_lanes('westbound'),
            'geometry': self._create_road_geometry('west'),
            'traffic_lights': ['tl_400_wb_straight', 'tl_400_wb_left'],
        })
        
        return roads
    
    def _create_chaharbag_lanes(self, direction: str) -> Dict:
        """خطوط چهارباغ: ۴ خط سواری + ۱ خط BRT + پیاده‌رو"""
        # جهت شمال به جنوب: خطوط مثبت = راست به چپ (خطي 1 = حاشیه راست)
        if direction == 'northbound':
            lanes = [
                LaneSpec(id=-1, type='sidewalk', width=self.SIDEWALK_WIDTH, speed=0),
                LaneSpec(id=-2, type='border', width=0.5, speed=0),
                LaneSpec(id=1, type='driving', width=self.LANE_WIDTH, speed=50),      # خط 1 (راست)
                LaneSpec(id=2, type='driving', width=self.LANE_WIDTH, speed=50),      # خط 2
                LaneSpec(id=3, type='driving', width=self.LANE_WIDTH, speed=50),      # خط 3
                LaneSpec(id=4, type='driving', width=self.BRT_LANE_WIDTH, speed=50,   # خط 4 (BRT)
                         access_restriction='bus'),
                LaneSpec(id=5, type='median', width=self.MEDIAN_WIDTH, speed=0),
            ]
        else:  # southbound
            lanes = [
                LaneSpec(id=5, type='median', width=self.MEDIAN_WIDTH, speed=0),
                LaneSpec(id=4, type='driving', width=self.BRT_LANE_WIDTH, speed=50,
                         access_restriction='bus'),
                LaneSpec(id=3, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=2, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=1, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=-2, type='border', width=0.5, speed=0),
                LaneSpec(id=-1, type='sidewalk', width=self.SIDEWALK_WIDTH, speed=0),
            ]
        return {'lane_offsets': lanes}
    
    def _create_azadi_lanes(self, direction: str) -> Dict:
        """خطوط آزادی: ۳ خط سواری + پیاده‌رو (بدون BRT)"""
        if direction == 'eastbound':
            lanes = [
                LaneSpec(id=-1, type='sidewalk', width=self.SIDEWALK_WIDTH, speed=0),
                LaneSpec(id=-2, type='border', width=0.5, speed=0),
                LaneSpec(id=1, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=2, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=3, type='driving', width=self.LANE_WIDTH, speed=50),  # خط چپ‌دوران
                LaneSpec(id=4, type='median', width=self.MEDIAN_WIDTH, speed=0),
            ]
        else:  # westbound
            lanes = [
                LaneSpec(id=4, type='median', width=self.MEDIAN_WIDTH, speed=0),
                LaneSpec(id=3, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=2, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=1, type='driving', width=self.LANE_WIDTH, speed=50),
                LaneSpec(id=-2, type='border', width=0.5, speed=0),
                LaneSpec(id=-1, type='sidewalk', width=self.SIDEWALK_WIDTH, speed=0),
            ]
        return {'lane_offsets': lanes}
    
    def _create_road_geometry(self, approach: str) -> List[Dict]:
        """هندسه مسیر (خط مستقیم به مرکز تقاطع)"""
        half_width = self.INTERSECTION_WIDTH / 2
        
        if approach == 'north':
            # از شمال به مرکز (y مثبت به 0)
            return [
                {'type': 'line', 'start': (0, self.APPROACH_LENGTH + half_width), 
                 'end': (0, half_width), 'heading': -math.pi/2}
            ]
        elif approach == 'south':
            # از جنوب به مرکز (y منفی به 0)
            return [
                {'type': 'line', 'start': (0, -self.APPROACH_LENGTH - half_width),
                 'end': (0, -half_width), 'heading': math.pi/2}
            ]
        elif approach == 'east':
            # از شرق به مرکز (x مثبت به 0)
            return [
                {'type': 'line', 'start': (self.APPROACH_LENGTH + half_width, 0),
                 'end': (half_width, 0), 'heading': math.pi}
            ]
        else:  # west
            return [
                {'type': 'line', 'start': (-self.APPROACH_LENGTH - half_width, 0),
                 'end': (-half_width, 0), 'heading': 0}
            ]
    
    def _define_junctions(self) -> List[Dict]:
        """تعریف تقاطع مرکزی"""
        half_iw = self.INTERSECTION_WIDTH / 2
        
        return [{
            'id': 1,
            'name': 'ChaharBagh_Azadi_Junction',
            'center': (0.0, 0.0),
            'connections': self._create_junction_connections(),
            'controllers': ['signal_controller_1'],
            'polygon': [
                (-half_iw, -half_iw), (half_iw, -half_iw),
                (half_iw, half_iw), (-half_iw, half_iw)
            ],
            'crosswalks': self._create_crosswalks(),
        }]
    
    def _create_junction_connections(self) -> List[Dict]:
        """اتصالات در تقاطع (منبع -> مقصد با خط مقصد)"""
        connections = []
        
        # شمال -> جنوب (مستقیم) - خطوط 1،2،3
        for lane_id in [1, 2, 3]:
            connections.append({
                'id': len(connections),
                'incoming_road': 100, 'connecting_road': 200,
                'contact_point': 'start',  # road 200 starts at junction
                'lane_link': [{'from': lane_id, 'to': lane_id}]
            })
        
        # شمال -> غرب (چپ‌دوران) - خط 4 (BRT) نمی‌تواند چپ بدورد
        connections.append({
            'id': len(connections),
            'incoming_road': 100, 'connecting_road': 400,
            'contact_point': 'end',  # road 400 ends at junction
            'lane_link': [{'from': 3, 'to': 3}]  # خط 3 شمال -> خط 3 غرب
        })
        
        # جنوب -> شمال (مستقیم)
        for lane_id in [1, 2, 3]:
            connections.append({
                'id': len(connections),
                'incoming_road': 200, 'connecting_road': 100,
                'contact_point': 'end',
                'lane_link': [{'from': lane_id, 'to': lane_id}]
            })
        
        # جنوب -> شرق (چپ‌دوران)
        connections.append({
            'id': len(connections),
            'incoming_road': 200, 'connecting_road': 300,
            'contact_point': 'start',
            'lane_link': [{'from': 3, 'to': 3}]
        })
        
        # شرق -> غرب (مستقیم) - خطوط 1،2
        for lane_id in [1, 2]:
            connections.append({
                'id': len(connections),
                'incoming_road': 300, 'connecting_road': 400,
                'contact_point': 'end',
                'lane_link': [{'from': lane_id, 'to': lane_id}]
            })
        
        # شرق -> شمال (چپ‌دوران) - خط 3
        connections.append({
            'id': len(connections),
            'incoming_road': 300, 'connecting_road': 100,
            'contact_point': 'start',
            'lane_link': [{'from': 3, 'to': 3}]
        })
        
        # غرب -> شرق (مستقیم)
        for lane_id in [1, 2]:
            connections.append({
                'id': len(connections),
                'incoming_road': 400, 'connecting_road': 300,
                'contact_point': 'start',
                'lane_link': [{'from': lane_id, 'to': lane_id}]
            })
        
        # غرب -> جنوب (چپ‌دوران) - خط 3
        connections.append({
            'id': len(connections),
            'incoming_road': 400, 'connecting_road': 200,
            'contact_point': 'end',
            'lane_link': [{'from': 3, 'to': 3}]
        })
        
        return connections
    
    def _create_crosswalks(self) -> List[Dict]:
        """پیاده‌روهای ۴ گوشه"""
        half_iw = self.INTERSECTION_WIDTH / 2
        cw_w = self.CROSSWALK_WIDTH
        
        return [
            {'id': 1, 'road': 100, 'position': half_iw - cw_w/2, 'width': cw_w, 'length': 2*half_iw},  # شمال
            {'id': 2, 'road': 200, 'position': -half_iw + cw_w/2, 'width': cw_w, 'length': 2*half_iw}, # جنوب
            {'id': 3, 'road': 300, 'position': half_iw - cw_w/2, 'width': cw_w, 'length': 2*half_iw},  # شرق
            {'id': 4, 'road': 400, 'position': -half_iw + cw_w/2, 'width': cw_w, 'length': 2*half_iw}, # غرب
        ]
    
    def _define_controllers(self) -> List[Dict]:
        """کنترلرهای سیگنال (Siemens Sitrafic compatible)"""
        return [{
            'id': 'signal_controller_1',
            'type': 'actuated',
            'sequence': [
                # فاز ۰: شمال-جنوب مستقیم
                {'id': 0, 'name': 'NB_SB_Straight', 'duration': 45, 'min': 20, 'max': 80,
                 'signals': ['tl_100_nb_straight', 'tl_200_sb_straight'],
                 'pedestrian': ['crosswalk_3', 'crosswalk_4']},  # شرق/غرب
                
                # فاز ۱: شمال-جنوب چپ
                {'id': 1, 'name': 'NB_SB_Left', 'duration': 15, 'min': 8, 'max': 25,
                 'signals': ['tl_100_nb_left', 'tl_200_sb_left'],
                 'pedestrian': []},
                
                # فاز ۲: شرق-غرب مستقیم
                {'id': 2, 'name': 'EB_WB_Straight', 'duration': 40, 'min': 15, 'max': 60,
                 'signals': ['tl_300_eb_straight', 'tl_400_wb_straight'],
                 'pedestrian': ['crosswalk_1', 'crosswalk_2']},  # شمال/جنوب
                
                # فاز ۳: شرق-غرب چپ
                {'id': 3, 'name': 'EB_WB_Left', 'duration': 12, 'min': 8, 'max': 20,
                 'signals': ['tl_300_eb_left', 'tl_400_wb_left'],
                 'pedestrian': []},
                
                # فاز ۴: همه قرمز + پیاده‌رو
                {'id': 4, 'name': 'All_Red_Pedestrian', 'duration': 20, 'min': 15, 'max': 30,
                 'signals': [],  # همه قرمز
                 'pedestrian': ['crosswalk_1', 'crosswalk_2', 'crosswalk_3', 'crosswalk_4']},
            ],
            'intergreen_matrix': self._create_intergreen_matrix(),
        }]
    
    def _create_intergreen_matrix(self) -> List[List[int]]:
        """ماتریس زمان بین فازها (ثانیه) - Станدارد ایران"""
        # ۵ فاز × ۵ فاز
        # ردیف = فاز فعلی، ستون = فاز بعدی
        return [
            [0, 5, 8, 8, 5],   # از فاز ۰
            [5, 0, 8, 5, 5],   # از فاز ۱
            [8, 8, 0, 5, 5],   # از فاز ۲
            [8, 5, 5, 0, 5],   # از فاز ۳
            [5, 5, 5, 5, 0],   # از فاز ۴
        ]
    
    def build_opendrive(self) -> str:
        """تولید فایل OpenDRIVE 1.7"""
        root = ET.Element('OpenDRIVE')
        root.set('version', '1.7')
        root.set('revMajor', '1')
        root.set('revMinor', '7')
        
        # Header
        header = ET.SubElement(root, 'header')
        header.set('name', 'Isfahan_ChaharBagh_Azadi')
        header.set('version', '1.0')
        header.set('date', '2024-01-15')
        header.set('north', '0.0')
        header.set('south', '0.0')
        header.set('east', '0.0')
        header.set('west', '0.0')
        header.set('vendor', 'Isfahan Municipality Traffic Engineering')
        
        # Roads
        for road_def in self.roads:
            road_elem = self._create_road_element(road_def)
            root.append(road_elem)
        
        # Junctions
        for junction_def in self.junctions:
            junction_elem = self._create_junction_element(junction_def)
            root.append(junction_elem)
        
        # Controllers
        for controller_def in self.controllers:
            controller_elem = self._create_controller_element(controller_def)
            root.append(controller_elem)
        
        # Pretty print
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def _create_road_element(self, road_def: Dict) -> ET.Element:
        road = ET.Element('road')
        road.set('name', road_def['name'])
        road.set('length', str(road_def['length']))
        road.set('id', str(road_def['id']))
        road.set('junction', str(road_def.get('junction', -1)))
        
        # Link (predecessor/successor)
        link = ET.SubElement(road, 'link')
        # Simplified - no predecessor/successor for standalone intersection
        
        # Type (speed limit)
        type_elem = ET.SubElement(road, 'type')
        type_elem.set('s', '0')
        type_elem.set('type', 'urban')
        speed_elem = ET.SubElement(type_elem, 'speed')
        speed_elem.set('max', '50')
        speed_elem.set('unit', 'km/h')
        
        # Plan View (Geometry)
        plan_view = ET.SubElement(road, 'planView')
        for i, geom in enumerate(road_def['geometry']):
            geometry = ET.SubElement(plan_view, 'geometry')
            geometry.set('s', '0')
            geometry.set('x', str(geom['start'][0]))
            geometry.set('y', str(geom['start'][1]))
            geometry.set('hdg', str(geom['heading']))
            geometry.set('length', str(self.APPROACH_LENGTH + self.INTERSECTION_WIDTH))
            
            line = ET.SubElement(geometry, 'line')
        
        # Elevation Profile (flat)
        elevation = ET.SubElement(road, 'elevationProfile')
        elev_elem = ET.SubElement(elevation, 'elevation')
        elev_elem.set('s', '0')
        elev_elem.set('a', '0')
        elev_elem.set('b', '0')
        elev_elem.set('c', '0')
        elev_elem.set('d', '0')
        
        # Lateral Profile (superelevation - none for intersection)
        lateral = ET.SubElement(road, 'lateralProfile')
        
        # Lanes
        lanes = ET.SubElement(road, 'lanes')
        
        # Lane Offset (for sidewalk/border)
        lane_offset = ET.SubElement(lanes, 'laneOffset')
        lane_offset.set('s', '0')
        lane_offset.set('a', '0')
        lane_offset.set('b', '0')
        lane_offset.set('c', '0')
        lane_offset.set('d', '0')
        
        # Lane Sections
        lane_section = ET.SubElement(lanes, 'laneSection')
        lane_section.set('s', '0')
        
        # Left lanes (positive IDs)
        left = ET.SubElement(lane_section, 'left')
        for lane_spec in road_def['lanes']['lane_offsets']:
            if lane_spec.id > 0:
                lane = self._create_lane_element(lane_spec)
                left.append(lane)
        
        # Center (none for divided roads)
        center = ET.SubElement(lane_section, 'center')
        # Lane 0 (reference)
        lane_zero = ET.SubElement(center, 'lane')
        lane_zero.set('id', '0')
        lane_zero.set('type', 'none')
        lane_zero.set('level', 'false')
        
        # Right lanes (negative IDs)
        right = ET.SubElement(lane_section, 'right')
        for lane_spec in road_def['lanes']['lane_offsets']:
            if lane_spec.id < 0:
                lane = self._create_lane_element(lane_spec)
                right.append(lane)
        
        return road
    
    def _create_lane_element(self, lane_spec: LaneSpec) -> ET.Element:
        lane = ET.Element('lane')
        lane.set('id', str(lane_spec.id))
        lane.set('type', lane_spec.type)
        lane.set('level', str(lane_spec.level).lower())
        
        # Width
        width = ET.SubElement(lane, 'width')
        width.set('sOffset', '0')
        width.set('s', '0')
        width.set('a', str(lane_spec.width))
        width.set('b', '0')
        width.set('c', '0')
        width.set('d', '0')
        
        # Speed
        if lane_spec.speed > 0:
            speed = ET.SubElement(lane, 'speed')
            speed.set('max', str(lane_spec.speed))
        
        # Access restriction
        if lane_spec.access_restriction != 'none':
            access = ET.SubElement(lane, 'access')
            access.set('restriction', lane_spec.access_restriction)
        
        # Lane markings
        if lane_spec.type == 'driving':
            # Right marking
            if lane_spec.id > 0:
                marking = ET.SubElement(lane, 'roadMark')
                marking.set('type', 'solid' if lane_spec.id == 1 else 'broken')
                marking.set('weight', 'standard')
                marking.set('color', 'white')
                marking.set('laneChange', 'increase' if lane_spec.id > 1 else 'none')
        
        return lane
    
    def _create_junction_element(self, junction_def: Dict) -> ET.Element:
        junction = ET.Element('junction')
        junction.set('name', junction_def['name'])
        junction.set('id', str(junction_def['id']))
        
        # Connections
        for conn in junction_def['connections']:
            connection = ET.SubElement(junction, 'connection')
            connection.set('id', str(conn['id']))
            connection.set('incomingRoad', str(conn['incoming_road']))
            connection.set('connectingRoad', str(conn['connecting_road']))
            connection.set('contactPoint', conn['contact_point'])
            
            for link in conn['lane_link']:
                lane_link = ET.SubElement(connection, 'laneLink')
                lane_link.set('from', str(link['from']))
                lane_link.set('to', str(link['to']))
        
        # Controllers
        for ctrl_id in junction_def['controllers']:
            controller = ET.SubElement(junction, 'controller')
            controller.set('id', ctrl_id)
            controller.set('type', 'signal')
            controller.set('sequence', '1')
        
        return junction
    
    def _create_controller_element(self, controller_def: Dict) -> ET.Element:
        controller = ET.Element('controller')
        controller.set('id', controller_def['id'])
        controller.set('type', controller_def['type'])
        controller.set('sequence', '1')
        
        control = ET.SubElement(controller, 'control')
        
        for phase in controller_def['sequence']:
            phase_elem = ET.SubElement(control, 'phase')
            phase_elem.set('name', phase['name'])
            phase_elem.set('duration', str(phase['duration']))
            phase_elem.set('minDuration', str(phase['min']))
            phase_elem.set('maxDuration', str(phase['max']))
            
            # Signals
            for sig in phase['signals']:
                signal = ET.SubElement(phase_elem, 'signal')
                signal.set('id', sig)
                signal.set('state', 'green')
            
            # Pedestrian
            for ped in phase.get('pedestrian', []):
                ped_signal = ET.SubElement(phase_elem, 'pedestrianSignal')
                ped_signal.set('id', ped)
                ped_signal.set('state', 'walk')
        
        # Intergreen matrix
        intergreen = ET.SubElement(controller, 'intergreen')
        for i, row in enumerate(controller_def['intergreen_matrix']):
            for j, val in enumerate(row):
                if i != j:
                    ig = ET.SubElement(intergreen, 'intergreenTime')
                    ig.set('from', str(i))
                    ig.set('to', str(j))
                    ig.set('time', str(val))
        
        return controller
    
    def save(self, filename: str = "Isfahan_ChaharBagh_Azadi.xodr"):
        """ذخیره فایل OpenDRIVE"""
        content = self.build_opendrive()
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ OpenDRIVE map saved to: {output_path}")
        
        # ذخیره متادیتا JSON برای استفاده در پایتون
        metadata = {
            'intersection': {
                'id': 'chaharbag_azadi',
                'name': 'ChaharBagh-Azadi',
                'coordinates': [32.6546, 51.6680],
                'roads': [r['id'] for r in self.roads],
                'junctions': [j['id'] for j in self.junctions],
                'controllers': [c['id'] for c in self.controllers],
                'camera_positions': self._get_camera_positions(),
            }
        }
        meta_path = self.output_dir / "isfahan_intersection_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Metadata saved to: {meta_path}")
        
        return output_path
    
    def _get_camera_positions(self) -> List[Dict]:
        """موقعیت‌های دوربین برای کالیبراسیون"""
        return [
            {"id": "cam_01", "location": (120.5, -45.2, 8.5), "rotation": (0, -90, 0), "fov": 90},
            {"id": "cam_02", "location": (-110.3, 42.1, 8.5), "rotation": (0, 90, 0), "fov": 90},
            {"id": "cam_03", "location": (45.8, 115.4, 8.5), "rotation": (0, 180, 0), "fov": 90},
            {"id": "cam_04", "location": (-48.2, -112.7, 8.5), "rotation": (0, 0, 0), "fov": 90},
            {"id": "cam_05", "location": (125.0, 50.0, 12.0), "rotation": (-15, -135, 0), "fov": 120},
            {"id": "cam_06", "location": (-115.0, -48.0, 12.0), "rotation": (-15, 45, 0), "fov": 120},
            {"id": "cam_07", "location": (50.0, -120.0, 12.0), "rotation": (-15, -45, 0), "fov": 120},
            {"id": "cam_08", "location": (-45.0, 125.0, 12.0), "rotation": (-15, 135, 0), "fov": 120},
        ]


def main():
    print("🏗️ Building Isfahan ChaharBagh-Azadi Intersection Map for CARLA...")
    builder = IsfahanIntersectionBuilder()
    output_path = builder.save()
    print(f"✅ Done! Import {output_path} in CARLA using:")
    print(f"   client.load_world('Isfahan_ChaharBagh_Azadi')")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import networkx as nx
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class SceneGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.objects = {}
        self.temporal_data = []
        self.spatial_thresholds = {
            "near": 0.5,      # meters
            "touching": 0.1,   # meters
            "on_table": 1.0    # height threshold
        }
    
    def load_tracking_and_3d_data(self, tracking_file: str, positions_3d_file: str):
        """Load tracking and 3D position data"""
        # Load tracking data
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                self.temporal_data = json.load(f)
        
        # Load 3D positions
        if os.path.exists(positions_3d_file):
            with open(positions_3d_file, 'r') as f:
                positions_3d = json.load(f)
            
            # Add objects to graph
            for track_id, obj_data in positions_3d.items():
                self.add_object(
                    obj_id=f"obj_{track_id}",
                    properties={
                        "track_id": int(track_id),
                        "class_name": obj_data["class_name"],
                        "position_3d": obj_data["position"],
                        "num_observations": obj_data["num_observations"]
                    }
                )
    
    def add_object(self, obj_id: str, properties: Dict[str, Any]):
        """Add object node to scene graph"""
        self.objects[obj_id] = properties
        self.graph.add_node(obj_id, **properties)
    
    def add_relationship(self, obj1_id: str, obj2_id: str, 
                        relation_type: str, properties: Dict = None):
        """Add relationship edge between objects"""
        if properties is None:
            properties = {}
        
        # Avoid duplicate edges
        if self.graph.has_edge(obj1_id, obj2_id):
            # Update existing edge with new properties
            self.graph[obj1_id][obj2_id].update(properties)
            if relation_type not in self.graph[obj1_id][obj2_id].get('relations', []):
                self.graph[obj1_id][obj2_id].setdefault('relations', []).append(relation_type)
        else:
            self.graph.add_edge(obj1_id, obj2_id, 
                               relation=relation_type, relations=[relation_type], **properties)
    
    def detect_spatial_relationships(self):
        """Detect spatial relationships between objects based on 3D positions"""
        object_positions = {}
        
        # Extract 3D positions
        for obj_id, properties in self.objects.items():
            if "position_3d" in properties:
                pos = np.array(properties["position_3d"])
                object_positions[obj_id] = pos
        
        # Detect pairwise spatial relationships
        for obj1_id, pos1 in object_positions.items():
            for obj2_id, pos2 in object_positions.items():
                if obj1_id != obj2_id:
                    self._analyze_spatial_relationship(obj1_id, obj2_id, pos1, pos2)
    
    def _analyze_spatial_relationship(self, obj1_id: str, obj2_id: str, 
                                    pos1: np.ndarray, pos2: np.ndarray):
        """Analyze spatial relationship between two objects"""
        distance_3d = np.linalg.norm(pos1 - pos2)
        distance_2d = np.linalg.norm(pos1[:2] - pos2[:2])  # X-Y plane distance
        height_diff = abs(pos1[2] - pos2[2])  # Z-axis difference
        
        relationships = []
        
        # Distance-based relationships
        if distance_3d < self.spatial_thresholds["touching"]:
            relationships.append("touching")
        elif distance_3d < self.spatial_thresholds["near"]:
            relationships.append("near")
        
        # Height-based relationships
        if height_diff > 0.1:  # Significant height difference
            if pos1[2] > pos2[2]:
                relationships.append("above")
            else:
                relationships.append("below")
        
        # Relative position relationships
        direction = pos2 - pos1
        if abs(direction[0]) > abs(direction[1]):  # X-direction dominant
            if direction[0] > 0:
                relationships.append("right_of")
            else:
                relationships.append("left_of")
        else:  # Y-direction dominant
            if direction[1] > 0:
                relationships.append("in_front_of")
            else:
                relationships.append("behind")
        
        # Add relationships to graph
        for relation in relationships:
            self.add_relationship(obj1_id, obj2_id, relation, {
                "distance_3d": float(distance_3d),
                "distance_2d": float(distance_2d),
                "height_diff": float(height_diff)
            })
    
    def detect_temporal_relationships(self):
        """Detect temporal relationships from tracking data"""
        # Track object interactions over time
        interaction_history = defaultdict(list)
        
        for frame_data in self.temporal_data:
            frame_id = frame_data["frame_id"]
            
            # Get object positions in this frame
            frame_objects = {}
            for detection in frame_data["detections"]:
                track_id = detection["track_id"]
                bbox = detection["bbox"]
                obj_id = f"obj_{track_id}"
                
                frame_objects[obj_id] = {
                    "bbox": bbox,
                    "class_name": detection["class_name"]
                }
            
            # Detect interactions between objects in this frame
            for obj1_id, obj1_data in frame_objects.items():
                for obj2_id, obj2_data in frame_objects.items():
                    if obj1_id != obj2_id:
                        # Check if objects are interacting (overlapping bboxes)
                        if self._bboxes_interact(obj1_data["bbox"], obj2_data["bbox"]):
                            interaction_history[(obj1_id, obj2_id)].append(frame_id)
        
        # Add temporal relationships based on interaction history
        for (obj1_id, obj2_id), frames in interaction_history.items():
            if len(frames) > 5:  # Sustained interaction
                self.add_relationship(obj1_id, obj2_id, "interacts_with", {
                    "interaction_frames": frames,
                    "interaction_duration": len(frames)
                })
    
    def _bboxes_interact(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes interact (overlap or are very close)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert center coordinates to corner coordinates
        left1, top1 = x1 - w1/2, y1 - h1/2
        right1, bottom1 = x1 + w1/2, y1 + h1/2
        left2, top2 = x2 - w2/2, y2 - h2/2
        right2, bottom2 = x2 + w2/2, y2 + h2/2
        
        # Check for overlap or close proximity
        overlap_x = max(0, min(right1, right2) - max(left1, left2))
        overlap_y = max(0, min(bottom1, bottom2) - max(top1, top2))
        
        # Consider interaction if overlapping or very close
        threshold = min(w1, w2, h1, h2) * 0.1  # 10% of smaller dimension
        
        return overlap_x > 0 and overlap_y > 0 or \
               (overlap_x == 0 and abs(right1 - left2) < threshold) or \
               (overlap_x == 0 and abs(right2 - left1) < threshold) or \
               (overlap_y == 0 and abs(bottom1 - top2) < threshold) or \
               (overlap_y == 0 and abs(bottom2 - top1) < threshold)
    
    def infer_scene_context(self):
        """Infer high-level scene context and add scene-level nodes"""
        # Analyze object classes to infer scene type
        object_classes = [props.get("class_name", "") for props in self.objects.values()]
        
        scene_type = "unknown"
        if any(cls in ["cup", "bottle", "bowl", "spoon"] for cls in object_classes):
            scene_type = "kitchen"
        elif any(cls in ["book", "laptop", "mouse"] for cls in object_classes):
            scene_type = "office"
        elif any(cls in ["bed", "pillow"] for cls in object_classes):
            scene_type = "bedroom"
        
        # Add scene node
        self.add_object("scene", {
            "type": "scene_context",
            "scene_type": scene_type,
            "num_objects": len(self.objects),
            "object_classes": list(set(object_classes))
        })
        
        # Connect all objects to scene
        for obj_id in self.objects.keys():
            if obj_id != "scene":
                self.add_relationship("scene", obj_id, "contains")
    
    def build_complete_graph(self, tracking_file: str, positions_3d_file: str, 
                           output_dir: str = "data/graphs"):
        """Build complete scene graph from all available data"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading data...")
        self.load_tracking_and_3d_data(tracking_file, positions_3d_file)
        
        print("Detecting spatial relationships...")
        self.detect_spatial_relationships()
        
        print("Detecting temporal relationships...")
        self.detect_temporal_relationships()
        
        print("Inferring scene context...")
        self.infer_scene_context()
        
        # Export graph in multiple formats
        output_base = os.path.join(output_dir, "scene_graph")
        
        # JSON format for easy parsing
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node_id, data in self.graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node_id,
                "properties": data
            })
        
        for src, dst, data in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": src,
                "target": dst,
                "properties": data
            })
        
        with open(f"{output_base}.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # GML format for NetworkX compatibility
        nx.write_gml(self.graph, f"{output_base}.gml")
        
        # GraphML format for visualization tools
        nx.write_graphml(self.graph, f"{output_base}.graphml")
        
        print(f"Scene graph saved to {output_base}.[json|gml|graphml]")
        print(f"Graph contains {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
        return self.graph
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the scene graph"""
        summary = {
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges),
            "object_classes": {},
            "relationship_types": {},
            "scene_description": self._generate_scene_description()
        }
        
        # Count object classes
        for node_id, data in self.graph.nodes(data=True):
            class_name = data.get("class_name", "unknown")
            summary["object_classes"][class_name] = summary["object_classes"].get(class_name, 0) + 1
        
        # Count relationship types
        for src, dst, data in self.graph.edges(data=True):
            relations = data.get("relations", [data.get("relation", "unknown")])
            for relation in relations:
                summary["relationship_types"][relation] = summary["relationship_types"].get(relation, 0) + 1
        
        return summary
    
    def _generate_scene_description(self) -> str:
        """Generate natural language description of the scene"""
        if "scene" in self.graph.nodes:
            scene_data = self.graph.nodes["scene"]
            scene_type = scene_data.get("scene_type", "unknown")
            num_objects = scene_data.get("num_objects", 0)
            object_classes = scene_data.get("object_classes", [])
            
            description = f"This appears to be a {scene_type} scene with {num_objects} objects. "
            description += f"The scene contains: {', '.join(object_classes)}. "
            
            # Add relationship information
            relationships = list(nx.get_edge_attributes(self.graph, "relation").values())
            if relationships:
                common_relations = list(set(relationships))
                description += f"Common spatial relationships include: {', '.join(common_relations[:3])}."
            
            return description
        
        return "Scene description not available."

def main():
    print("Scene Graph Builder Module")
    
    builder = SceneGraphBuilder()
    
    # Check for required input files
    tracking_file = "data/tracking/tracking_results.json"
    positions_file = "data/3d_points/object_3d_positions.json"
    
    if os.path.exists(tracking_file) and os.path.exists(positions_file):
        print("Building scene graph from tracking and 3D data...")
        graph = builder.build_complete_graph(tracking_file, positions_file)
        
        # Print summary
        summary = builder.get_graph_summary()
        print("\nScene Graph Summary:")
        print(f"Objects: {summary['num_nodes']}")
        print(f"Relationships: {summary['num_edges']}")
        print(f"Object classes: {summary['object_classes']}")
        print(f"Relationship types: {summary['relationship_types']}")
        print(f"Description: {summary['scene_description']}")
        
    else:
        print("Required input files not found:")
        print(f"Tracking: {tracking_file} ({'✓' if os.path.exists(tracking_file) else '✗'})")
        print(f"3D positions: {positions_file} ({'✓' if os.path.exists(positions_file) else '✗'})")

if __name__ == "__main__":
    main()
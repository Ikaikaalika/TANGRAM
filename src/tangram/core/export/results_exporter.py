#!/usr/bin/env python3
"""
Results Export and Evaluation System

This module provides comprehensive export and evaluation capabilities
for the TANGRAM robotic scene understanding pipeline.

Features:
- Generate detailed HTML reports with visualizations
- Export videos of simulation and tracking results
- Create JSON logs for quantitative analysis  
- Performance metrics and evaluation statistics
- Integration with all pipeline components

Author: TANGRAM Team
License: MIT
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from jinja2 import Template

from config import RESULTS_DIR, EXPORTS_DIR, VIDEOS_DIR, LOGS_DIR
from utils.logging_utils import setup_logger, log_function_call
from utils.file_utils import save_json, ensure_directory
from utils.video_utils import extract_video_info

logger = setup_logger(__name__)

class ResultsExporter:
    """
    Comprehensive results export and evaluation system.
    
    Handles export of pipeline results in multiple formats including
    HTML reports, videos, JSON logs, and performance metrics.
    """
    
    def __init__(self, experiment_name: str = None):
        """
        Initialize results exporter.
        
        Args:
            experiment_name: Name for this experiment run
        """
        self.experiment_name = experiment_name or f"tangram_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.export_dir = EXPORTS_DIR / self.experiment_name
        
        # Create export directories
        ensure_directory(self.export_dir)
        ensure_directory(self.export_dir / "videos")
        ensure_directory(self.export_dir / "images")
        ensure_directory(self.export_dir / "data")
        
        # Results storage
        self.pipeline_results = {}
        self.performance_metrics = {}
        self.error_log = []
        
        logger.info(f"Initialized results exporter for experiment: {self.experiment_name}")
    
    @log_function_call()
    def collect_pipeline_results(self, data_dir: Path = None) -> Dict[str, Any]:
        """
        Collect results from all pipeline components.
        
        Args:
            data_dir: Data directory to search for results (None = use default)
            
        Returns:
            Dictionary containing all collected results
        """
        if data_dir is None:
            data_dir = Path("data")
        
        results = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "tracking_results": None,
            "segmentation_results": None,
            "reconstruction_results": None,
            "scene_graph": None,
            "llm_interpretation": None,
            "simulation_results": None,
            "performance_metrics": {}
        }
        
        try:
            # Load tracking results
            tracking_file = data_dir / "tracking" / "tracking_results.json"
            if tracking_file.exists():
                results["tracking_results"] = self._load_json_safe(tracking_file)
                logger.info("Loaded tracking results")
            
            # Load segmentation results
            segmentation_file = data_dir / "masks" / "segmentation_results.json"
            if segmentation_file.exists():
                results["segmentation_results"] = self._load_json_safe(segmentation_file)
                logger.info("Loaded segmentation results")
            
            # Load 3D reconstruction results
            reconstruction_file = data_dir / "3d_points" / "object_3d_positions.json"
            if reconstruction_file.exists():
                results["reconstruction_results"] = self._load_json_safe(reconstruction_file)
                logger.info("Loaded 3D reconstruction results")
            
            # Load scene graph
            scene_graph_file = data_dir / "graphs" / "scene_graph.json"
            if scene_graph_file.exists():
                results["scene_graph"] = self._load_json_safe(scene_graph_file)
                logger.info("Loaded scene graph")
            
            # Load LLM interpretation
            llm_file = data_dir / "graphs" / "llm_interpretation.json"
            if llm_file.exists():
                results["llm_interpretation"] = self._load_json_safe(llm_file)
                logger.info("Loaded LLM interpretation")
            
            # Load simulation results
            simulation_file = data_dir / "simulation" / "simulation_results.json"
            if simulation_file.exists():
                results["simulation_results"] = self._load_json_safe(simulation_file)
                logger.info("Loaded simulation results")
            
            self.pipeline_results = results
            
            # Calculate performance metrics
            self.performance_metrics = self._calculate_performance_metrics(results)
            results["performance_metrics"] = self.performance_metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to collect pipeline results: {e}")
            self.error_log.append(f"Pipeline collection error: {e}")
            return results
    
    def _load_json_safe(self, file_path: Path) -> Optional[Dict]:
        """Safely load JSON file with error handling."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            self.error_log.append(f"JSON load error ({file_path}): {e}")
            return None
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics from pipeline results."""
        metrics = {
            "pipeline_completion": {},
            "object_detection": {},
            "tracking_quality": {},
            "reconstruction_quality": {},
            "task_execution": {},
            "overall_performance": 0.0
        }
        
        try:
            # Pipeline completion metrics
            completed_steps = 0
            total_steps = 6  # Total number of pipeline steps
            
            if results["tracking_results"]:
                completed_steps += 1
                metrics["pipeline_completion"]["tracking"] = True
            if results["segmentation_results"]:
                completed_steps += 1
                metrics["pipeline_completion"]["segmentation"] = True
            if results["reconstruction_results"]:
                completed_steps += 1
                metrics["pipeline_completion"]["reconstruction"] = True
            if results["scene_graph"]:
                completed_steps += 1
                metrics["pipeline_completion"]["scene_graph"] = True
            if results["llm_interpretation"]:
                completed_steps += 1
                metrics["pipeline_completion"]["llm_interpretation"] = True
            if results["simulation_results"]:
                completed_steps += 1
                metrics["pipeline_completion"]["simulation"] = True
            
            metrics["pipeline_completion"]["completion_rate"] = completed_steps / total_steps
            
            # Object detection metrics
            if results["tracking_results"]:
                tracking_data = results["tracking_results"]
                total_frames = len(tracking_data)
                frames_with_detections = len([f for f in tracking_data if f.get("detections")])
                
                metrics["object_detection"]["total_frames"] = total_frames
                metrics["object_detection"]["frames_with_detections"] = frames_with_detections
                metrics["object_detection"]["detection_rate"] = frames_with_detections / max(total_frames, 1)
                
                # Count unique objects
                unique_objects = set()
                for frame in tracking_data:
                    for detection in frame.get("detections", []):
                        unique_objects.add(detection.get("track_id"))
                
                metrics["object_detection"]["unique_objects_tracked"] = len(unique_objects)
            
            # Task execution metrics
            if results["simulation_results"]:
                sim_data = results["simulation_results"]
                task_results = sim_data.get("task_results", [])
                
                if task_results:
                    successful_tasks = len([t for t in task_results if t.get("success", False)])
                    total_tasks = len(task_results)
                    
                    metrics["task_execution"]["total_tasks"] = total_tasks
                    metrics["task_execution"]["successful_tasks"] = successful_tasks
                    metrics["task_execution"]["success_rate"] = successful_tasks / max(total_tasks, 1)
                    
                    # Average task duration
                    durations = [t.get("duration", 0) for t in task_results]
                    metrics["task_execution"]["average_duration"] = np.mean(durations) if durations else 0
            
            # Overall performance score (weighted average)
            weights = {
                "completion": 0.3,
                "detection": 0.2,
                "task_execution": 0.5
            }
            
            overall_score = 0.0
            overall_score += weights["completion"] * metrics["pipeline_completion"]["completion_rate"]
            
            if "detection_rate" in metrics["object_detection"]:
                overall_score += weights["detection"] * metrics["object_detection"]["detection_rate"]
            
            if "success_rate" in metrics["task_execution"]:
                overall_score += weights["task_execution"] * metrics["task_execution"]["success_rate"]
            
            metrics["overall_performance"] = overall_score
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            self.error_log.append(f"Metrics calculation error: {e}")
        
        return metrics
    
    @log_function_call()
    def generate_html_report(self, include_images: bool = True) -> Path:
        """
        Generate comprehensive HTML report.
        
        Args:
            include_images: Whether to generate and include visualization images
            
        Returns:
            Path to generated HTML report
        """
        try:
            if include_images:
                self._generate_visualization_images()
            
            # Generate HTML content
            html_content = self._create_html_content()
            
            # Save HTML report
            report_path = self.export_dir / "report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            self.error_log.append(f"HTML report error: {e}")
            return None
    
    def _generate_visualization_images(self):
        """Generate visualization images for the report."""
        images_dir = self.export_dir / "images"
        
        try:
            # Performance metrics chart
            self._create_performance_chart(images_dir / "performance_metrics.png")
            
            # Object detection timeline
            if self.pipeline_results.get("tracking_results"):
                self._create_detection_timeline(images_dir / "detection_timeline.png")
            
            # Task execution results
            if self.pipeline_results.get("simulation_results"):
                self._create_task_execution_chart(images_dir / "task_execution.png")
            
            # Scene graph visualization
            if self.pipeline_results.get("scene_graph"):
                self._create_scene_graph_visualization(images_dir / "scene_graph.png")
            
        except Exception as e:
            logger.error(f"Failed to generate visualization images: {e}")
            self.error_log.append(f"Visualization error: {e}")
    
    def _create_performance_chart(self, output_path: Path):
        """Create performance metrics bar chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pipeline completion chart
        completion_data = self.performance_metrics.get("pipeline_completion", {})
        steps = []
        status = []
        
        for step, completed in completion_data.items():
            if step != "completion_rate":
                steps.append(step.replace("_", " ").title())
                status.append("Completed" if completed else "Failed")
        
        if steps:
            colors = ['green' if s == "Completed" else 'red' for s in status]
            ax1.barh(steps, [1] * len(steps), color=colors, alpha=0.7)
            ax1.set_xlabel("Status")
            ax1.set_title("Pipeline Step Completion")
            ax1.set_xlim(0, 1)
        
        # Performance metrics
        metrics_data = {
            "Overall Performance": self.performance_metrics.get("overall_performance", 0),
            "Detection Rate": self.performance_metrics.get("object_detection", {}).get("detection_rate", 0),
            "Task Success Rate": self.performance_metrics.get("task_execution", {}).get("success_rate", 0)
        }
        
        metrics_names = list(metrics_data.keys())
        metrics_values = list(metrics_data.values())
        
        bars = ax2.bar(metrics_names, metrics_values, color=['blue', 'orange', 'green'], alpha=0.7)
        ax2.set_ylabel("Score")
        ax2.set_title("Performance Metrics")
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detection_timeline(self, output_path: Path):
        """Create object detection timeline chart."""
        tracking_data = self.pipeline_results["tracking_results"]
        
        frame_ids = []
        detection_counts = []
        
        for frame_data in tracking_data:
            frame_ids.append(frame_data["frame_id"])
            detection_counts.append(len(frame_data.get("detections", [])))
        
        plt.figure(figsize=(12, 6))
        plt.plot(frame_ids, detection_counts, marker='o', linewidth=2, markersize=4)
        plt.xlabel("Frame Number")
        plt.ylabel("Number of Detections")
        plt.title("Object Detection Timeline")
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        avg_detections = np.mean(detection_counts)
        plt.axhline(y=avg_detections, color='red', linestyle='--', 
                   label=f'Average: {avg_detections:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_task_execution_chart(self, output_path: Path):
        """Create task execution results chart."""
        sim_data = self.pipeline_results["simulation_results"]
        task_results = sim_data.get("task_results", [])
        
        if not task_results:
            return
        
        # Task success/failure counts
        task_types = {}
        for task in task_results:
            task_type = task.get("type", "unknown")
            if task_type not in task_types:
                task_types[task_type] = {"success": 0, "failure": 0}
            
            if task.get("success", False):
                task_types[task_type]["success"] += 1
            else:
                task_types[task_type]["failure"] += 1
        
        # Create stacked bar chart
        types = list(task_types.keys())
        successes = [task_types[t]["success"] for t in types]
        failures = [task_types[t]["failure"] for t in types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stacked bar chart
        width = 0.6
        ax1.bar(types, successes, width, label='Success', color='green', alpha=0.7)
        ax1.bar(types, failures, width, bottom=successes, label='Failure', color='red', alpha=0.7)
        
        ax1.set_ylabel('Number of Tasks')
        ax1.set_title('Task Execution Results by Type')
        ax1.legend()
        
        # Task duration histogram
        durations = [task.get("duration", 0) for task in task_results]
        ax2.hist(durations, bins=10, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_ylabel('Number of Tasks')
        ax2.set_title('Task Duration Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scene_graph_visualization(self, output_path: Path):
        """Create scene graph visualization."""
        try:
            import networkx as nx
            
            scene_data = self.pipeline_results["scene_graph"]
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in scene_data.get("nodes", []):
                node_id = node["id"]
                properties = node.get("properties", {})
                G.add_node(node_id, **properties)
            
            # Add edges
            for edge in scene_data.get("edges", []):
                source = edge["source"]
                target = edge["target"]
                properties = edge.get("properties", {})
                G.add_edge(source, target, **properties)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Use spring layout for nice positioning
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            node_colors = []
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                if node_data.get("type") == "scene_context":
                    node_colors.append('red')
                else:
                    node_colors.append('lightblue')
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=1500, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, alpha=0.6)
            
            # Draw labels
            labels = {}
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                class_name = node_data.get("class_name", node_id)
                labels[node_id] = class_name
            
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            # Draw edge labels
            edge_labels = {}
            for edge in G.edges(data=True):
                source, target, data = edge
                relation = data.get("relation", "")
                if relation:
                    edge_labels[(source, target)] = relation
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
            
            plt.title("Scene Graph Visualization")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            logger.warning("NetworkX not available for scene graph visualization")
        except Exception as e:
            logger.error(f"Failed to create scene graph visualization: {e}")
    
    def _create_html_content(self) -> str:
        """Create HTML report content."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TANGRAM Pipeline Results - {{ experiment_name }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }
        h3 { color: #7f8c8d; }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .status-success { color: #27ae60; }
        .status-failure { color: #e74c3c; }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        .error-log {
            background-color: #fff5f5;
            border: 1px solid #feb2b2;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }
        .json-data {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>TANGRAM Pipeline Results</h1>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ experiment_name }}</div>
                <div>Experiment Name</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ timestamp }}</div>
                <div>Generated At</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {{ 'status-success' if overall_performance > 0.7 else 'status-failure' }}">
                    {{ "%.1f%%" | format(overall_performance * 100) }}
                </div>
                <div>Overall Performance</div>
            </div>
        </div>
        
        {% if performance_metrics %}
        <h2>Performance Metrics</h2>
        <div class="image-container">
            <img src="images/performance_metrics.png" alt="Performance Metrics" />
        </div>
        
        <table>
            <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
            <tr>
                <td>Pipeline Completion Rate</td>
                <td>{{ "%.1f%%" | format(performance_metrics.pipeline_completion.completion_rate * 100) }}</td>
                <td class="{{ 'status-success' if performance_metrics.pipeline_completion.completion_rate > 0.8 else 'status-failure' }}">
                    {{ 'Good' if performance_metrics.pipeline_completion.completion_rate > 0.8 else 'Needs Improvement' }}
                </td>
            </tr>
            {% if performance_metrics.object_detection.detection_rate is defined %}
            <tr>
                <td>Object Detection Rate</td>
                <td>{{ "%.1f%%" | format(performance_metrics.object_detection.detection_rate * 100) }}</td>
                <td class="{{ 'status-success' if performance_metrics.object_detection.detection_rate > 0.7 else 'status-failure' }}">
                    {{ 'Good' if performance_metrics.object_detection.detection_rate > 0.7 else 'Needs Improvement' }}
                </td>
            </tr>
            {% endif %}
            {% if performance_metrics.task_execution.success_rate is defined %}
            <tr>
                <td>Task Success Rate</td>
                <td>{{ "%.1f%%" | format(performance_metrics.task_execution.success_rate * 100) }}</td>
                <td class="{{ 'status-success' if performance_metrics.task_execution.success_rate > 0.8 else 'status-failure' }}">
                    {{ 'Good' if performance_metrics.task_execution.success_rate > 0.8 else 'Needs Improvement' }}
                </td>
            </tr>
            {% endif %}
        </table>
        {% endif %}
        
        {% if tracking_results %}
        <h2>Object Detection & Tracking</h2>
        <div class="image-container">
            <img src="images/detection_timeline.png" alt="Detection Timeline" />
        </div>
        
        <p>
            <strong>Total Frames Processed:</strong> {{ tracking_results | length }}<br>
            <strong>Unique Objects Tracked:</strong> {{ performance_metrics.object_detection.unique_objects_tracked if performance_metrics.object_detection.unique_objects_tracked is defined else 'N/A' }}
        </p>
        {% endif %}
        
        {% if simulation_results %}
        <h2>Robot Task Execution</h2>
        <div class="image-container">
            <img src="images/task_execution.png" alt="Task Execution Results" />
        </div>
        
        <p>
            <strong>Total Tasks:</strong> {{ simulation_results.task_results | length }}<br>
            <strong>Successful Tasks:</strong> {{ performance_metrics.task_execution.successful_tasks if performance_metrics.task_execution.successful_tasks is defined else 'N/A' }}<br>
            <strong>Average Duration:</strong> {{ "%.2f seconds" | format(performance_metrics.task_execution.average_duration) if performance_metrics.task_execution.average_duration is defined else 'N/A' }}
        </p>
        {% endif %}
        
        {% if scene_graph %}
        <h2>Scene Graph</h2>
        <div class="image-container">
            <img src="images/scene_graph.png" alt="Scene Graph Visualization" />
        </div>
        
        <p>
            <strong>Objects:</strong> {{ scene_graph.nodes | length }}<br>
            <strong>Relationships:</strong> {{ scene_graph.edges | length }}
        </p>
        {% endif %}
        
        {% if llm_interpretation %}
        <h2>LLM Interpretation</h2>
        <p><strong>Scene Explanation:</strong></p>
        <p>{{ llm_interpretation.scene_explanation if llm_interpretation.scene_explanation else 'No explanation available' }}</p>
        
        {% if llm_interpretation.task_sequence %}
        <p><strong>Generated Tasks:</strong></p>
        <table>
            <tr><th>Task ID</th><th>Type</th><th>Description</th><th>Duration (est.)</th></tr>
            {% for task in llm_interpretation.task_sequence %}
            <tr>
                <td>{{ task.id if task.id is defined else 'N/A' }}</td>
                <td>{{ task.type if task.type else 'N/A' }}</td>
                <td>{{ task.description if task.description else 'N/A' }}</td>
                <td>{{ task.estimated_duration if task.estimated_duration is defined else 'N/A' }}s</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        {% endif %}
        
        {% if error_log %}
        <h2>Error Log</h2>
        <div class="error-log">
            {% for error in error_log %}
            <p>• {{ error }}</p>
            {% endfor %}
        </div>
        {% endif %}
        
        <h2>Raw Data</h2>
        <p>Complete pipeline results are available in the exported JSON files:</p>
        <ul>
            <li><a href="data/complete_results.json">Complete Results (JSON)</a></li>
            <li><a href="data/performance_metrics.json">Performance Metrics (JSON)</a></li>
        </ul>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            Generated by TANGRAM Robotic Scene Understanding Pipeline
        </footer>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        
        return template.render(
            experiment_name=self.experiment_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_performance=self.performance_metrics.get("overall_performance", 0),
            performance_metrics=self.performance_metrics,
            tracking_results=self.pipeline_results.get("tracking_results"),
            segmentation_results=self.pipeline_results.get("segmentation_results"),
            reconstruction_results=self.pipeline_results.get("reconstruction_results"),
            scene_graph=self.pipeline_results.get("scene_graph"),
            llm_interpretation=self.pipeline_results.get("llm_interpretation"),
            simulation_results=self.pipeline_results.get("simulation_results"),
            error_log=self.error_log
        )
    
    @log_function_call()
    def export_json_logs(self) -> List[Path]:
        """Export detailed JSON logs."""
        exported_files = []
        data_dir = self.export_dir / "data"
        
        try:
            # Export complete results
            complete_results_path = data_dir / "complete_results.json"
            save_json(self.pipeline_results, complete_results_path)
            exported_files.append(complete_results_path)
            
            # Export performance metrics
            metrics_path = data_dir / "performance_metrics.json"
            save_json(self.performance_metrics, metrics_path)
            exported_files.append(metrics_path)
            
            # Export error log
            if self.error_log:
                error_log_path = data_dir / "error_log.json"
                save_json({"errors": self.error_log, "timestamp": datetime.now().isoformat()}, 
                         error_log_path)
                exported_files.append(error_log_path)
            
            logger.info(f"Exported {len(exported_files)} JSON log files")
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export JSON logs: {e}")
            return exported_files
    
    @log_function_call()
    def create_summary_video(self, input_video_path: str = None) -> Optional[Path]:
        """Create summary video with tracking overlays."""
        if not input_video_path:
            return None
            
        try:
            tracking_data = self.pipeline_results.get("tracking_results")
            if not tracking_data:
                logger.warning("No tracking data available for video creation")
                return None
            
            output_path = self.export_dir / "videos" / "summary_video.mp4"
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Find tracking data for this frame
                frame_data = None
                for data in tracking_data:
                    if data["frame_id"] == frame_idx:
                        frame_data = data
                        break
                
                # Draw tracking overlays
                if frame_data:
                    for detection in frame_data.get("detections", []):
                        # Get bounding box
                        bbox = detection["bbox"]
                        x, y, w, h = bbox
                        
                        # Convert center coordinates to corner coordinates
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"ID:{detection['track_id']} {detection['class_name']}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add frame info
                info_text = f"Frame: {frame_idx}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
            logger.info(f"Created summary video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create summary video: {e}")
            return None
    
    @log_function_call()
    def export_complete_report(self, input_video_path: str = None) -> Dict[str, Any]:
        """Export complete results package."""
        logger.info("Starting complete results export...")
        
        # Collect all pipeline results
        results = self.collect_pipeline_results()
        
        # Generate HTML report
        html_report = self.generate_html_report()
        
        # Export JSON logs
        json_files = self.export_json_logs()
        
        # Create summary video
        video_file = None
        if input_video_path:
            video_file = self.create_summary_video(input_video_path)
        
        export_summary = {
            "experiment_name": self.experiment_name,
            "export_directory": str(self.export_dir),
            "html_report": str(html_report) if html_report else None,
            "json_files": [str(f) for f in json_files],
            "summary_video": str(video_file) if video_file else None,
            "performance_score": self.performance_metrics.get("overall_performance", 0),
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Save export summary
        summary_path = self.export_dir / "export_summary.json"
        save_json(export_summary, summary_path)
        
        logger.info(f"Complete results exported to: {self.export_dir}")
        logger.info(f"Overall performance score: {export_summary['performance_score']:.2f}")
        
        return export_summary

def main():
    """Test the results export system."""
    print("TANGRAM Results Export and Evaluation System")
    
    # Create exporter
    exporter = ResultsExporter("test_experiment")
    
    # Export results (will use any available data)
    export_summary = exporter.export_complete_report()
    
    print(f"Export completed: {export_summary['export_directory']}")
    print(f"Performance score: {export_summary['performance_score']:.2f}")

if __name__ == "__main__":
    main()
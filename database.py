"""
Database module for Space Station Safety Detection System
Handles detection logs, alerts, and analytics storage
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

class DetectionDatabase:
    """SQLite database for detection logging and analytics"""
    
    def __init__(self, db_path: str = "detection_logs.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Detection logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                class_name TEXT,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                image_width INTEGER,
                image_height INTEGER,
                model_used TEXT
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                severity TEXT,
                title TEXT,
                message TEXT,
                class_name TEXT,
                confidence REAL,
                acknowledged BOOLEAN DEFAULT 0,
                acknowledged_at DATETIME,
                acknowledged_by TEXT
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                fps REAL,
                processing_time_ms REAL,
                detections_count INTEGER,
                cpu_usage REAL,
                memory_usage REAL,
                gpu_usage REAL
            )
        """)
        
        # Sessions table (for tracking monitoring sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                total_frames INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                total_alerts INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_detection(self, source: str, class_name: str, confidence: float,
                     bbox: tuple, image_size: tuple, model: str = "YOLOv8") -> int:
        """Log a single detection"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO detections 
            (source, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
             image_width, image_height, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (source, class_name, confidence, bbox[0], bbox[1], bbox[2], bbox[3],
              image_size[0], image_size[1], model))
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return detection_id
    
    def log_alert(self, severity: str, title: str, message: str,
                  class_name: Optional[str] = None, confidence: Optional[float] = None) -> int:
        """Log an alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (severity, title, message, class_name, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (severity, title, message, class_name, confidence))
        
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return alert_id
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "admin"):
        """Mark an alert as acknowledged"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts 
            SET acknowledged = 1, 
                acknowledged_at = CURRENT_TIMESTAMP,
                acknowledged_by = ?
            WHERE id = ?
        """, (acknowledged_by, alert_id))
        
        conn.commit()
        conn.close()
    
    def log_system_metrics(self, fps: float, processing_time: float, 
                          detections_count: int, cpu: float = 0.0, 
                          memory: float = 0.0, gpu: float = 0.0):
        """Log system performance metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_metrics 
            (fps, processing_time_ms, detections_count, cpu_usage, memory_usage, gpu_usage)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (fps, processing_time, detections_count, cpu, memory, gpu))
        
        conn.commit()
        conn.close()
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """Get recent detections"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM detections 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_unacknowledged_alerts(self, severity: Optional[str] = None) -> List[Dict]:
        """Get unacknowledged alerts"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if severity:
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE acknowledged = 0 AND severity = ?
                ORDER BY timestamp DESC
            """, (severity,))
        else:
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE acknowledged = 0
                ORDER BY timestamp DESC
            """)
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_detection_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get detection statistics for the last N hours"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute("""
            SELECT COUNT(*) FROM detections
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
        """, (hours,))
        total = cursor.fetchone()[0]
        
        # By class
        cursor.execute("""
            SELECT class_name, COUNT(*) as count
            FROM detections
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            GROUP BY class_name
            ORDER BY count DESC
        """, (hours,))
        by_class = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute("""
            SELECT AVG(confidence) FROM detections
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
        """, (hours,))
        avg_conf = cursor.fetchone()[0] or 0.0
        
        # Alerts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM alerts
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            GROUP BY severity
        """, (hours,))
        alerts_by_severity = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_detections': total,
            'by_class': by_class,
            'average_confidence': avg_conf,
            'alerts_by_severity': alerts_by_severity
        }
    
    def get_detections_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """Get detections as pandas DataFrame"""
        conn = self.get_connection()
        
        query = """
            SELECT * FROM detections
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(hours,))
        conn.close()
        
        return df
    
    def get_alerts_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """Get alerts as pandas DataFrame"""
        conn = self.get_connection()
        
        query = """
            SELECT * FROM alerts
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(hours,))
        conn.close()
        
        return df
    
    def export_to_csv(self, table: str, output_path: str, hours: int = 24):
        """Export table data to CSV"""
        conn = self.get_connection()
        
        query = f"""
            SELECT * FROM {table}
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(hours,))
        df.to_csv(output_path, index=False)
        
        conn.close()
        return output_path
    
    def clear_old_data(self, days: int = 30):
        """Clear data older than N days"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM detections
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        cursor.execute("""
            DELETE FROM alerts
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        cursor.execute("""
            DELETE FROM system_metrics
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def get_database_size(self) -> float:
        """Get database file size in MB"""
        path = Path(self.db_path)
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0

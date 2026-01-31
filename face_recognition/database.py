#!/usr/bin/env python3
"""
Face Recognition Database - SQLite storage for face encodings
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

DB_PATH = Path(__file__).parent / "faces.db"

def get_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize database schema"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # People table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Face encodings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            image_path TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    ''')
    
    # Sightings/logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            confidence REAL,
            image_path TEXT,
            location TEXT,
            seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES people(id)
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_encodings_person ON face_encodings(person_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sightings_person ON sightings(person_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sightings_time ON sightings(seen_at)')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")

def add_person(name: str, notes: str = "") -> int:
    """Add a new person to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO people (name, notes) VALUES (?, ?)',
        (name, notes)
    )
    person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id

def get_person(person_id: int) -> Optional[dict]:
    """Get person by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM people WHERE id = ?', (person_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_person_by_name(name: str) -> Optional[dict]:
    """Get person by name"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM people WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def list_people() -> List[dict]:
    """List all people"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, COUNT(fe.id) as encoding_count 
        FROM people p 
        LEFT JOIN face_encodings fe ON p.id = fe.person_id 
        GROUP BY p.id
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_encoding(person_id: int, encoding: np.ndarray, image_path: str = None, confidence: float = None) -> int:
    """Add face encoding for a person"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Convert numpy array to bytes
    encoding_bytes = encoding.tobytes()
    
    cursor.execute(
        'INSERT INTO face_encodings (person_id, encoding, image_path, confidence) VALUES (?, ?, ?, ?)',
        (person_id, encoding_bytes, image_path, confidence)
    )
    encoding_id = cursor.lastrowid
    
    # Update person's updated_at
    cursor.execute(
        'UPDATE people SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
        (person_id,)
    )
    
    conn.commit()
    conn.close()
    return encoding_id

def get_all_encodings() -> List[Tuple[int, str, np.ndarray]]:
    """Get all encodings with person info for matching"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT fe.person_id, p.name, fe.encoding 
        FROM face_encodings fe 
        JOIN people p ON fe.person_id = p.id
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        encoding = np.frombuffer(row['encoding'], dtype=np.float64)
        results.append((row['person_id'], row['name'], encoding))
    
    return results

def log_sighting(person_id: int, confidence: float, image_path: str = None, location: str = None):
    """Log a face sighting"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO sightings (person_id, confidence, image_path, location) VALUES (?, ?, ?, ?)',
        (person_id, confidence, image_path, location)
    )
    conn.commit()
    conn.close()

def get_recent_sightings(limit: int = 20) -> List[dict]:
    """Get recent sightings"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.*, p.name 
        FROM sightings s 
        LEFT JOIN people p ON s.person_id = p.id 
        ORDER BY s.seen_at DESC 
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_person(person_id: int) -> bool:
    """Delete a person and their encodings"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM people WHERE id = ?', (person_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# Initialize on import
if __name__ == "__main__":
    init_database()
    print("\n📊 Database ready!")
    print(f"   People: {len(list_people())}")

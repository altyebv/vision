"""
SQLite database for storing transactions.
Lightweight and flexible schema.
FIXED: Added missing methods for API integration.
"""
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from models import ExtractionResult, ValidationIssue


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str = "receipts.db"):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Main transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                receipt_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                overall_confidence REAL,
                needs_review INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Transaction fields (flexible for any field)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transaction_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                raw_value TEXT,
                ocr_confidence REAL,
                manually_edited INTEGER DEFAULT 0,
                edited_at TIMESTAMP,
                FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
            )
        """)

        # Validation issues
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                field_name TEXT,
                severity TEXT,
                message TEXT,
                resolved INTEGER DEFAULT 0,
                FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
            )
        """)

        # Known entities (for autocomplete)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                value TEXT NOT NULL,
                display_name TEXT,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified INTEGER DEFAULT 0,
                UNIQUE(entity_type, value)
            )
        """)

        # Processing stats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE DEFAULT CURRENT_DATE,
                total_processed INTEGER DEFAULT 0,
                auto_verified INTEGER DEFAULT 0,
                manually_reviewed INTEGER DEFAULT 0,
                avg_processing_time REAL,
                avg_confidence REAL,
                UNIQUE(date)
            )
        """)

        # Create indices for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transaction_fields_transaction_id 
            ON transaction_fields(transaction_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transaction_fields_field_name 
            ON transaction_fields(field_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_known_entities_type_value 
            ON known_entities(entity_type, value)
        """)

        self.conn.commit()
        print(f"✓ Database initialized: {self.db_path}")

    def save_result(self, result: ExtractionResult) -> int:
        """
        Save extraction result to database.

        Args:
            result: ExtractionResult to save

        Returns:
            Transaction ID
        """
        cursor = self.conn.cursor()

        # Insert transaction
        cursor.execute("""
            INSERT INTO transactions 
            (filename, receipt_type, processed_at, overall_confidence, needs_review, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            result.filename,
            result.receipt_type.value,
            datetime.now(),
            result.overall_confidence,
            1 if result.needs_review else 0,
            'pending' if result.needs_review else 'verified'
        ))

        transaction_id = cursor.lastrowid

        # Insert fields
        for field_name in ['transaction_id', 'datetime', 'from_account',
                           'to_account', 'receiver_name', 'comment', 'amount']:
            field = getattr(result.data, field_name, None)

            if field:
                cursor.execute("""
                    INSERT INTO transaction_fields
                    (transaction_id, field_name, field_value, raw_value, ocr_confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    transaction_id,
                    field_name,
                    field.value,
                    field.raw_text,
                    field.confidence
                ))

                # Update known entities (for autocomplete)
                if field_name in ['from_account', 'to_account']:
                    self._update_known_entity(field_name, field.value, None)
                elif field_name == 'receiver_name':
                    # Link name with to_account for better autocomplete
                    to_account = getattr(result.data, 'to_account', None)
                    account_number = to_account.value if to_account else None
                    self._update_known_entity('receiver_name', field.value, account_number)

        # Insert validation issues
        for issue in result.issues:
            cursor.execute("""
                INSERT INTO validation_issues
                (transaction_id, field_name, severity, message)
                VALUES (?, ?, ?, ?)
            """, (
                transaction_id,
                issue.field,
                issue.severity,
                issue.message
            ))

        # Update stats
        self._update_stats(result)

        self.conn.commit()

        return transaction_id

    def save_batch(self, results: List[ExtractionResult]) -> List[int]:
        """
        Save multiple results.

        Args:
            results: List of ExtractionResult

        Returns:
            List of transaction IDs
        """
        ids = []
        for result in results:
            transaction_id = self.save_result(result)
            ids.append(transaction_id)

        return ids

    def get_transaction(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """Get transaction by ID with all fields."""
        cursor = self.conn.cursor()

        # Get transaction
        cursor.execute("""
            SELECT * FROM transactions WHERE id = ?
        """, (transaction_id,))

        row = cursor.fetchone()
        if not row:
            return None

        transaction = dict(row)

        # Get fields
        cursor.execute("""
            SELECT * FROM transaction_fields WHERE transaction_id = ?
        """, (transaction_id,))

        fields = {}
        for field_row in cursor.fetchall():
            field_dict = dict(field_row)
            fields[field_dict['field_name']] = field_dict

        transaction['fields'] = fields

        # Get issues
        cursor.execute("""
            SELECT * FROM validation_issues WHERE transaction_id = ?
        """, (transaction_id,))

        issues = [dict(row) for row in cursor.fetchall()]
        transaction['issues'] = issues

        return transaction

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all transactions that need review."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT id FROM transactions 
            WHERE needs_review = 1 AND status = 'pending'
            ORDER BY created_at DESC
        """)

        transactions = []
        for row in cursor.fetchall():
            transaction = self.get_transaction(row['id'])
            if transaction:
                transactions.append(transaction)

        return transactions

    def get_receipts(self, days: int = 7, include_pending: bool = True) -> List[Dict[str, Any]]:
        """
        Get receipts for export.

        Args:
            days: Number of days to look back
            include_pending: Whether to include pending reviews

        Returns:
            List of receipt records with flattened fields
        """
        cursor = self.conn.cursor()

        # Build query
        query = """
            SELECT 
                t.id,
                t.filename,
                t.receipt_type,
                t.created_at,
                t.overall_confidence,
                t.needs_review,
                t.status
            FROM transactions t
            WHERE t.created_at >= datetime('now', '-' || ? || ' days')
        """

        if not include_pending:
            query += " AND t.needs_review = 0"

        query += " ORDER BY t.created_at DESC"

        cursor.execute(query, (days,))
        transactions = [dict(row) for row in cursor.fetchall()]

        # Get fields for each transaction
        results = []
        for trans in transactions:
            cursor.execute("""
                SELECT field_name, field_value, ocr_confidence
                FROM transaction_fields
                WHERE transaction_id = ?
            """, (trans['id'],))

            # Flatten fields
            record = trans.copy()
            for field_row in cursor.fetchall():
                field_name = field_row['field_name']
                record[field_name] = field_row['field_value']
                record[f'{field_name}_confidence'] = field_row['ocr_confidence']

            results.append(record)

        return results

    def mark_reviewed(self, transaction_id: int):
        """Mark transaction as reviewed."""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE transactions 
            SET status = 'reviewed', needs_review = 0
            WHERE id = ?
        """, (transaction_id,))

        self.conn.commit()

    def update_field(self, transaction_id: int, field_name: str, new_value: str):
        """Update a single field value (manual edit)."""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE transaction_fields
            SET field_value = ?, manually_edited = 1, edited_at = ?
            WHERE transaction_id = ? AND field_name = ?
        """, (new_value, datetime.now(), transaction_id, field_name))

        self.conn.commit()

        # Update known entity
        if field_name in ['from_account', 'to_account', 'receiver_name']:
            # For receiver_name, get associated account
            account_number = None
            if field_name == 'receiver_name':
                cursor.execute("""
                    SELECT field_value FROM transaction_fields
                    WHERE transaction_id = ? AND field_name = 'to_account'
                """, (transaction_id,))
                row = cursor.fetchone()
                if row:
                    account_number = row['field_value']

            self._update_known_entity(field_name, new_value, account_number, verified=True)

    def update_receipt(self, transaction_id: int, data: Dict[str, Any]):
        """
        Update multiple fields in a receipt (for API).

        Args:
            transaction_id: Transaction ID
            data: Dictionary of field updates {field_name: new_value}
        """
        cursor = self.conn.cursor()

        for field_name, new_value in data.items():
            if field_name in ['transaction_id', 'datetime', 'from_account',
                            'to_account', 'receiver_name', 'comment', 'amount']:
                self.update_field(transaction_id, field_name, new_value)

        # Mark as reviewed
        self.mark_reviewed(transaction_id)

    def delete_receipt(self, transaction_id: int):
        """
        Delete a receipt and all associated data.

        Args:
            transaction_id: Transaction ID to delete
        """
        cursor = self.conn.cursor()

        # Delete transaction (CASCADE will handle fields and issues)
        cursor.execute("""
            DELETE FROM transactions WHERE id = ?
        """, (transaction_id,))

        self.conn.commit()

    def check_duplicate(self, transaction_id: str) -> bool:
        """
        Check if transaction ID already exists.

        Args:
            transaction_id: Transaction ID to check

        Returns:
            True if duplicate exists
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) as count
            FROM transaction_fields
            WHERE field_name = 'transaction_id' AND field_value = ?
        """, (transaction_id,))

        row = cursor.fetchone()
        return row['count'] > 0

    def get_known_entities(self, entity_type: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get known entities for autocomplete.

        Args:
            entity_type: Type of entity ('from_account', 'to_account', 'receiver_name')
            limit: Maximum number of results

        Returns:
            List of entity dictionaries with 'value' and 'display_name'
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT value, display_name, frequency, verified
            FROM known_entities
            WHERE entity_type = ?
            ORDER BY verified DESC, frequency DESC, last_seen DESC
            LIMIT ?
        """, (entity_type, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'value': row['value'],
                'display_name': row['display_name'] or row['value'],
                'frequency': row['frequency'],
                'verified': bool(row['verified'])
            })

        return results

    def search_entities(self, entity_type: str, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Search for entities matching query.

        Args:
            entity_type: Type of entity
            query: Search query
            limit: Maximum results

        Returns:
            List of matching entities
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT value, display_name, frequency, verified
            FROM known_entities
            WHERE entity_type = ? 
            AND (value LIKE ? OR display_name LIKE ?)
            ORDER BY verified DESC, frequency DESC, last_seen DESC
            LIMIT ?
        """, (entity_type, f'%{query}%', f'%{query}%', limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'value': row['value'],
                'display_name': row['display_name'] or row['value'],
                'frequency': row['frequency'],
                'verified': bool(row['verified'])
            })

        return results

    def query_transactions(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query transactions with filters.

        Args:
            filters: Dictionary of filters
                - from_account: Account number
                - to_account: Account number
                - receiver_name: Receiver name
                - date_from: Start date
                - date_to: End date
                - min_amount: Minimum amount
                - max_amount: Maximum amount

        Returns:
            List of matching transactions
        """
        cursor = self.conn.cursor()

        # Build dynamic query
        query = """
            SELECT DISTINCT t.id
            FROM transactions t
            LEFT JOIN transaction_fields tf ON t.id = tf.transaction_id
            WHERE 1=1
        """
        params = []

        # Add filters
        if filters.get('from_account'):
            query += """ AND t.id IN (
                SELECT transaction_id FROM transaction_fields
                WHERE field_name = 'from_account' AND field_value LIKE ?
            )"""
            params.append(f'%{filters["from_account"]}%')

        if filters.get('to_account'):
            query += """ AND t.id IN (
                SELECT transaction_id FROM transaction_fields
                WHERE field_name = 'to_account' AND field_value LIKE ?
            )"""
            params.append(f'%{filters["to_account"]}%')

        if filters.get('receiver_name'):
            query += """ AND t.id IN (
                SELECT transaction_id FROM transaction_fields
                WHERE field_name = 'receiver_name' AND field_value LIKE ?
            )"""
            params.append(f'%{filters["receiver_name"]}%')

        if filters.get('date_from'):
            query += " AND t.created_at >= ?"
            params.append(filters['date_from'])

        if filters.get('date_to'):
            query += " AND t.created_at <= ?"
            params.append(filters['date_to'])

        query += " ORDER BY t.created_at DESC"

        cursor.execute(query, params)
        transaction_ids = [row['id'] for row in cursor.fetchall()]

        # Get full transaction data
        transactions = []
        for trans_id in transaction_ids:
            trans = self.get_transaction(trans_id)
            if trans:
                # Apply amount filters if specified
                if filters.get('min_amount') or filters.get('max_amount'):
                    amount_field = trans['fields'].get('amount')
                    if amount_field:
                        try:
                            amount = float(amount_field['field_value'].replace(',', ''))
                            if filters.get('min_amount') and amount < float(filters['min_amount']):
                                continue
                            if filters.get('max_amount') and amount > float(filters['max_amount']):
                                continue
                        except ValueError:
                            pass

                transactions.append(trans)

        return transactions

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get processing statistics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary of statistics
        """
        cursor = self.conn.cursor()

        # Get from processing_stats table
        cursor.execute("""
            SELECT 
                SUM(total_processed) as total,
                SUM(auto_verified) as auto_verified,
                SUM(manually_reviewed) as manually_reviewed,
                AVG(avg_confidence) as avg_confidence,
                AVG(avg_processing_time) as avg_time
            FROM processing_stats
            WHERE date >= date('now', '-' || ? || ' days')
        """, (days,))

        row = cursor.fetchone()

        # Calculate total amount
        cursor.execute("""
            SELECT SUM(CAST(REPLACE(field_value, ',', '') AS REAL)) as total_amount
            FROM transaction_fields tf
            JOIN transactions t ON tf.transaction_id = t.id
            WHERE tf.field_name = 'amount'
            AND t.created_at >= datetime('now', '-' || ? || ' days')
        """, (days,))

        amount_row = cursor.fetchone()

        return {
            'total_processed': int(row['total'] or 0),
            'auto_verified': int(row['auto_verified'] or 0),
            'manually_reviewed': int(row['manually_reviewed'] or 0),
            'avg_confidence': float(row['avg_confidence'] or 0.0),
            'avg_processing_time': float(row['avg_time'] or 0.0),
            'total_amount': float(amount_row['total_amount'] or 0.0)
        }

    def _update_known_entity(self, entity_type: str, value: str,
                            associated_value: Optional[str] = None,
                            verified: bool = False):
        """
        Update known entity frequency and associations.

        Args:
            entity_type: Type of entity
            value: Entity value
            associated_value: Associated value (e.g., account for name)
            verified: Whether manually verified
        """
        if entity_type not in ['from_account', 'to_account', 'receiver_name']:
            return

        cursor = self.conn.cursor()

        # For receiver_name, store account number as display_name for linking
        display_name = associated_value if entity_type == 'receiver_name' else None

        cursor.execute("""
            INSERT INTO known_entities (entity_type, value, display_name, verified)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_type, value) DO UPDATE SET
                frequency = frequency + 1,
                last_seen = CURRENT_TIMESTAMP,
                display_name = COALESCE(excluded.display_name, display_name),
                verified = MAX(verified, excluded.verified)
        """, (entity_type, value, display_name, 1 if verified else 0))

        self.conn.commit()

    def _update_stats(self, result: ExtractionResult):
        """Update daily processing statistics."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO processing_stats 
            (date, total_processed, auto_verified, manually_reviewed,
             avg_processing_time, avg_confidence)
            VALUES (CURRENT_DATE, 1, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_processed = total_processed + 1,
                auto_verified = auto_verified + excluded.auto_verified,
                manually_reviewed = manually_reviewed + excluded.manually_reviewed,
                avg_processing_time = (avg_processing_time * total_processed + excluded.avg_processing_time) / (total_processed + 1),
                avg_confidence = (avg_confidence * total_processed + excluded.avg_confidence) / (total_processed + 1)
        """, (
            0 if result.needs_review else 1,
            1 if result.needs_review else 0,
            result.processing_time,
            result.overall_confidence
        ))

        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup."""
        self.close()
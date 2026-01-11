"""
FastAPI server for receipt processing.
Separate from main.py CLI interface.
"""
import config  # Import first to fix OpenMP
import io
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import pandas as pd

from processor import ReceiptProcessor
from database import Database
from models import ExtractionResult, ReceiptType, ValidationIssue

# Initialize
app = FastAPI(
    title="Receipt OCR API",
    version="1.0.0",
    description="Automated receipt data extraction system"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor (lazy loading)
processor: Optional[ReceiptProcessor] = None
database: Optional[Database] = None


def get_processor():
    """Get or initialize processor."""
    global processor
    if processor is None:
        print("Initializing processor...")
        processor = ReceiptProcessor(ocr_engine="auto")
    return processor


def get_database():
    """Get or initialize database."""
    global database
    if database is None:
        print("Initializing database...")
        database = Database("receipts.db")
    return database


# === Response Models ===

class FieldData(BaseModel):
    value: str
    confidence: float
    raw_text: str
    needs_review: bool


class IssueData(BaseModel):
    field: str
    severity: str
    message: str


class ExtractionResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    receipt_type: str
    data: dict
    issues: List[IssueData]
    overall_confidence: float
    needs_review: bool
    processing_time: float


class BatchResponse(BaseModel):
    total: int
    successful: int
    needs_review: int
    avg_confidence: float
    results: List[ExtractionResponse]


class StatsResponse(BaseModel):
    total_processed: int
    auto_verified: int
    manually_reviewed: int
    avg_confidence: float
    total_amount: Optional[float] = None


# === Helper Functions ===

def convert_result_to_response(result: ExtractionResult, db_id: Optional[int] = None) -> ExtractionResponse:
    """Convert ExtractionResult to API response."""
    data_dict = {}

    for field_name in ['transaction_id', 'datetime', 'from_account',
                       'to_account', 'receiver_name', 'comment', 'amount']:
        field_result = getattr(result.data, field_name, None)
        if field_result:
            data_dict[field_name] = {
                'value': field_result.value,
                'confidence': field_result.confidence,
                'raw_text': field_result.raw_text,
                'needs_review': field_result.needs_review
            }
        else:
            data_dict[field_name] = None

    issues_list = [
        IssueData(
            field=issue.field,
            severity=issue.severity,
            message=issue.message
        )
        for issue in result.issues
    ]

    return ExtractionResponse(
        id=db_id,
        filename=result.filename,
        receipt_type=result.receipt_type.value,
        data=data_dict,
        issues=issues_list,
        overall_confidence=result.overall_confidence,
        needs_review=result.needs_review,
        processing_time=result.processing_time
    )


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Cannot decode image")

    return image


# === API Endpoints ===

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Receipt OCR API",
        "version": "1.0.0",
        "endpoints": {
            "extract": "/api/extract",
            "batch": "/api/extract/batch",
            "pending": "/api/pending",
            "stats": "/api/stats",
            "export": "/api/export/excel",
            "docs": "/docs"
        }
    }


@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_receipt(
        file: UploadFile = File(...),
        save_to_db: bool = Query(True, description="Save result to database")
):
    """
    Extract data from a single receipt image.

    Args:
        file: Receipt image (PNG, JPG, JPEG)
        save_to_db: Whether to save to database

    Returns:
        Extracted transaction data with confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")

        # Load image
        image = load_image_from_upload(file)

        # Get processor
        proc = get_processor()

        # Save temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename

        cv2.imwrite(str(temp_path), image)

        # Process
        result = proc.process(temp_path)

        # Save to database
        db_id = None
        if save_to_db:
            db = get_database()
            db_id = db.save_result(result)

        # Cleanup
        temp_path.unlink(missing_ok=True)

        return convert_result_to_response(result, db_id)

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/api/extract/batch", response_model=BatchResponse)
async def extract_batch(
        files: List[UploadFile] = File(...),
        save_to_db: bool = Query(True, description="Save results to database")
):
    """
    Extract data from multiple receipt images.

    Args:
        files: List of receipt images (max 50)
        save_to_db: Whether to save to database

    Returns:
        Batch processing results with statistics
    """
    try:
        if len(files) > 50:
            raise HTTPException(400, "Maximum 50 files per batch")

        # Get processor
        proc = get_processor()

        # Create temp directory
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        # Save all files
        temp_paths = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue

            temp_path = temp_dir / file.filename

            # Save file
            with open(temp_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)

            temp_paths.append(temp_path)

        # Process batch
        results = proc.process_batch(temp_paths)

        # Save to database
        db_ids = []
        if save_to_db:
            db = get_database()
            db_ids = db.save_batch(results)

        # Convert results
        response_results = [
            convert_result_to_response(r, db_id)
            for r, db_id in zip(results, db_ids if db_ids else [None] * len(results))
        ]

        # Cleanup
        for path in temp_paths:
            path.unlink(missing_ok=True)

        # Calculate stats
        successful = sum(1 for r in results if not r.needs_review)
        needs_review = len(results) - successful
        avg_conf = sum(r.overall_confidence for r in results) / len(results) if results else 0.0

        return BatchResponse(
            total=len(results),
            successful=successful,
            needs_review=needs_review,
            avg_confidence=avg_conf,
            results=response_results
        )

    except Exception as e:
        raise HTTPException(500, f"Batch processing failed: {str(e)}")


@app.post("/api/validate")
async def validate_extraction(data: dict):
    """
    Validate extracted data without processing image.

    Args:
        data: Extracted field data

    Returns:
        Validation results
    """
    from validator import Validator

    try:
        validator = Validator()

        # Reconstruct TransactionData from dict
        from models import TransactionData, FieldResult

        field_results = {}
        for field_name, field_data in data.items():
            if field_data:
                field_results[field_name] = FieldResult(
                    value=field_data['value'],
                    confidence=field_data['confidence'],
                    raw_text=field_data.get('raw_text', ''),
                    needs_review=field_data.get('needs_review', False)
                )

        transaction_data = TransactionData(**field_results)

        # Validate
        result = validator.validate(
            transaction_data,
            ReceiptType.UNKNOWN,
            "manual_validation",
            0.0
        )

        return convert_result_to_response(result)

    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")


@app.get("/api/pending", response_model=List[ExtractionResponse])
async def get_pending_reviews():
    """Get all receipts pending manual review."""
    try:
        db = get_database()
        pending = db.get_pending_reviews()

        # Convert to response format
        results = []
        for record in pending:
            # Reconstruct result from DB record
            from models import TransactionData, FieldResult

            field_data = {}
            for field in ['transaction_id', 'datetime', 'from_account',
                          'to_account', 'receiver_name', 'comment', 'amount']:
                value = record.get(field)
                if value:
                    field_data[field] = FieldResult(
                        value=value,
                        confidence=record.get(f'{field}_confidence', 0.0),
                        raw_text=value,
                        needs_review=True
                    )

            result = ExtractionResult(
                filename=record['filename'],
                receipt_type=ReceiptType[record['receipt_type']],
                data=TransactionData(**field_data),
                issues=[],
                overall_confidence=record['overall_confidence'],
                needs_review=True,
                processing_time=0.0
            )

            results.append(convert_result_to_response(result, record['id']))

        return results

    except Exception as e:
        raise HTTPException(500, f"Failed to fetch pending reviews: {str(e)}")


@app.put("/api/receipts/{receipt_id}")
async def update_receipt(receipt_id: int, data: dict):
    """
    Update receipt data after manual review.

    Args:
        receipt_id: Database ID
        data: Updated field values

    Returns:
        Success message
    """
    try:
        db = get_database()

        # Update in database
        db.update_receipt(receipt_id, data)

        return {"status": "ok", "message": "Receipt updated", "id": receipt_id}

    except Exception as e:
        raise HTTPException(500, f"Update failed: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(days: int = Query(30, description="Number of days to include")):
    """
    Get processing statistics.

    Args:
        days: Number of days to include (default: 30)

    Returns:
        Statistics summary
    """
    try:
        db = get_database()
        stats = db.get_stats(days=days)

        return StatsResponse(**stats)

    except Exception as e:
        raise HTTPException(500, f"Failed to fetch stats: {str(e)}")


@app.get("/api/export/excel")
async def export_excel(
        days: int = Query(7, description="Number of days to include"),
        include_pending: bool = Query(True, description="Include pending reviews")
):
    """
    Export receipts to Excel file.

    Args:
        days: Number of days to include
        include_pending: Whether to include pending reviews

    Returns:
        Excel file download
    """
    try:
        db = get_database()

        # Fetch data
        receipts = db.get_receipts(days=days, include_pending=include_pending)

        if not receipts:
            raise HTTPException(404, "No receipts found")

        # Convert to DataFrame
        df = pd.DataFrame(receipts)

        # Reorder columns
        columns = [
            'id', 'filename', 'receipt_type', 'transaction_id', 'datetime',
            'from_account', 'to_account', 'receiver_name', 'comment',
            'amount', 'overall_confidence', 'needs_review', 'created_at'
        ]
        df = df[[col for col in columns if col in df.columns]]

        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Receipts')

        output.seek(0)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"receipts_{timestamp}.xlsx"

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")


@app.delete("/api/receipts/{receipt_id}")
async def delete_receipt(receipt_id: int):
    """Delete a receipt from database."""
    try:
        db = get_database()
        db.delete_receipt(receipt_id)

        return {"status": "ok", "message": "Receipt deleted", "id": receipt_id}

    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")


# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# === Startup/Shutdown ===

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("=" * 60)
    print("Receipt OCR API Server")
    print("=" * 60)
    print(f"Starting up...")
    print(f"API docs: http://localhost:8000/docs")
    print(f"Interactive docs: http://localhost:8000/redoc")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global database
    if database:
        database.close()
    print("\nServer shutdown complete.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
from app.models import Request

async def get_request_stats(db: AsyncSession):
    stmt = select(Request.execution_time, Request.size)
    result = await db.execute(stmt)
    df = pd.DataFrame(result.all(), columns=["execution_time", "size"])

    if df.empty:
        return {"error": "Нет данных"}

    stats = {
        "count": len(df),
        "execution_time": {
            "mean_ms": round(float(df["execution_time"].mean()), 2),
            "median_ms":  int(df["execution_time"].median()),
            "p50_ms":   int(df["execution_time"].quantile(0.50)),
            "p95_ms":   int(df["execution_time"].quantile(0.95)),
            "p99_ms":   int(df["execution_time"].quantile(0.99)),
            "max_ms":   int(df["execution_time"].max()),
            "min_ms":   int(df["execution_time"].min()),
        },
        "size_bytes": {
            "mean":     round(float(df["size"].mean()), 1),
            "median":   int(df["size"].median()),
            "p95":      int(df["size"].quantile(0.95)),
            "p99":      int(df["size"].quantile(0.99)),
            "max":      int(df["size"].max()),
        }
    }

    return stats

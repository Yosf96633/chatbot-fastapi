from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore  # ← new
from langchain_openai import OpenAIEmbeddings               # ← new
import app.services.chat.graph as graph_module
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is required!")

    async with AsyncConnectionPool(
        conninfo=db_url,
        min_size=2,
        max_size=10,
        kwargs={"autocommit": True},
    ) as pool:

        # your existing checkpointer — no change
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        # ← new store using same Neon pool
        store = AsyncPostgresStore(
            pool,
            index={
                "embed": OpenAIEmbeddings(),
                "dims": 1536,
            }
        )
        await store.setup()  # creates store tables + vector columns

        # pass both to build_graph
        graph_module.workflow = graph_module.build_graph(checkpointer, store)

        async def keepalive():
            while True:
                await asyncio.sleep(240)
                try:
                    async with pool.connection() as conn:
                        await conn.execute("SELECT 1")
                except Exception:
                    pass

        task = asyncio.create_task(keepalive())
        yield
        task.cancel()


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3001", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes.chat import router
app.include_router(router)
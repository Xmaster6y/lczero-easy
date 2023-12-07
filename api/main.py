"""
Main API module.
"""

import logging
import subprocess
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api.schema import ErrorDetail, SuccessDetail


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger("uvicorn")
    logger.info("Starting up...")
    mkdir_cmd = ["mkdir", "-p", "tmp"]
    result = subprocess.run(
        mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        logger.error(result.stderr.decode("utf-8"))
    yield
    rmdir_cmd = ["rm", "-r", "tmp"]
    result = subprocess.run(
        rmdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        logger.error(result.stderr.decode("utf-8"))
    logger.info("Shutting down...")


def clean_files(directory="*"):
    logger = logging.getLogger("uvicorn")
    rmdir_cmd = ["rm", "-r", f"tmp/{directory}"]
    result = subprocess.run(
        rmdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        logger.error(result.stderr.decode("utf-8"))


app = FastAPI(
    title="LeelaChessZero Easy API",
    lifespan=lifespan,
)


@app.get(
    "/",
    tags=["HOME"],
    summary="Home page",
    status_code=200,
    response_model=SuccessDetail,
)
async def home():
    """
    Home page.
    """
    return {"success": "Welcome to the API!"}


@app.post(
    "/lc0/convert",
    tags=["LC0"],
    summary="Convert a LeelaChessZero model to an ONNX model",
    status_code=200,
    response_model=SuccessDetail,
    responses={
        400: {"model": ErrorDetail},
        500: {"model": ErrorDetail},
    },
)
async def convert(
    network_file: Annotated[UploadFile, "Network file to convert"],
    background_tasks: BackgroundTasks,
):
    """
    Convert a LeelaChessZero model to an ONNX model.
    """
    dir_id = uuid.uuid4()
    directory = f"tmp/{str(dir_id)}"
    mkdir_cmd = ["mkdir", "-p", directory]
    result = subprocess.run(
        mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": result.stderr.decode("utf-8"),
            },
        )
    filename = "model.gz"
    with open(f"{directory}/{filename}", "wb") as buffer:
        buffer.write(await network_file.read())
    cmd = ["file", f"{directory}/{filename}"]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": result.stderr.decode("utf-8"),
            },
        )
    stdout = result.stdout.decode("utf-8")
    file_type = stdout.split(":")[1].strip().split(",")[0]
    if file_type == "gzip compressed data":
        original_filename = stdout.split('"')[1]
        cmd = ["gzip", f"{directory}/{filename}", "-d", "-N"]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={
                    "error": result.stderr.decode("utf-8"),
                },
            )
    elif file_type == "data":
        original_filename = filename
    else:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid file type: {file_type}",
            },
        )
    original_filename_noext = original_filename.split(".")[0]
    cmd = [
        "lc0",
        "leela2onnx",
        "--onnx2pytorch",
        f"--input={directory}/{original_filename}",
        f"--output={directory}/{original_filename_noext}.onnx",
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": result.stderr.decode("utf-8"),
            },
        )
    stdout = result.stdout.decode("utf-8")
    with open(f"{directory}/{original_filename_noext}.txt", "w") as buffer:
        buffer.write(stdout)

    cmd = [
        "zip",
        "-r",
        f"{directory}/{original_filename_noext}.zip",
        f"{directory}/{original_filename_noext}.onnx",
        f"{directory}/{original_filename_noext}.txt",
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": result.stderr.decode("utf-8"),
            },
        )
    background_tasks.add_task(clean_files, directory=dir_id)
    return FileResponse(
        f"{directory}/{original_filename_noext}.zip",
        media_type="application/zip",
        filename=f"{original_filename_noext}.zip",
    )

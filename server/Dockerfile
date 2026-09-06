# Stage 1: Build C++ pose_solver_cpp voi libs cua container
# g2o duoc build tu source vi Debian libg2o-dev thieu cmake config files.
# Pin sang tag 20230806_git (= v0.2.0) de dam bao cache on dinh va khop voi
# native/libs/ dung cho chay local (ngoai Docker).
###############################################################################
FROM python:3.10-slim AS cpp-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config git \
        libopencv-dev \
        libeigen3-dev \
        libsuitesparse-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build g2o tu source, pin tag cu the de layer cache khong bi invalidate.
RUN git clone --depth 1 --branch 20230806_git \
        https://github.com/RainerKuemmerle/g2o.git /tmp/g2o \
    && cmake -S /tmp/g2o -B /tmp/g2o/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DG2O_BUILD_EXAMPLES=OFF \
        -DG2O_BUILD_APPS=OFF \
        -DG2O_USE_OPENGL=OFF \
    && cmake --build /tmp/g2o/build --parallel "$(( $(nproc) / 2 < 2 ? 2 : $(nproc) / 2 ))" \
    && cmake --install /tmp/g2o/build \
    && sed -i '/find_dependency.*OpenGL/d' /usr/local/lib/cmake/g2o/g2oConfig.cmake \
    && ldconfig \
    && rm -rf /tmp/g2o

RUN pip install pybind11[global]

WORKDIR /build
COPY native/pose_solver_cpp/ ./pose_solver_cpp/

RUN cmake -S pose_solver_cpp -B pose_solver_cpp/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build pose_solver_cpp/build --parallel "$(( $(nproc) / 2 < 2 ? 2 : $(nproc) / 2 ))"

###############################################################################
# Stage 2: Runtime
###############################################################################
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
        libopencv-dev libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy g2o runtime libs tu builder stage
COPY --from=cpp-builder /usr/local/lib/libg2o* /usr/local/lib/
RUN ldconfig

# Layer 1: core deps (cache bi invalidate it hon)
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Layer 2: AI/ML deps nang (torch CPU ~250MB, cache rieng de khong rebuild lai khi sua core)
COPY requirements-ai.txt ./
RUN pip install -r requirements-ai.txt

COPY . .

# Copy .so tu builder stage — link voi dung libs cua container
COPY --from=cpp-builder /build/pose_solver_cpp/build/pose_solver_cpp*.so \
    /app/app/modules/artifact_pose/

# Do not run the API as root. Runtime data is supplied through /app/data and
# /app/model volumes; make the mount points writable by the service account.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/model \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]

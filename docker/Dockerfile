# Build the context emitted by docker/prepare.py --game-dir PATH.
FROM python:3.11-slim AS wheels
WORKDIR /src
COPY pyproject.toml setup.py LICENSE README.md ./
COPY src/wc3env/ src/wc3env/
COPY wc3agent/ wc3agent/
COPY wc3hook/ wc3hook/
RUN python -m pip wheel --no-deps --wheel-dir /wheels/environment . \
    && python -m pip wheel --no-deps --wheel-dir /wheels/agent ./wc3agent \
    && python -m pip download --no-deps --only-binary=:all: --platform win_amd64 --python-version 3.11 \
       --dest /wheels/environment "numpy>=1.24,<3"

FROM ubuntu:24.04 AS environment
ARG WINE_VERSION=11.0.0.0~noble-1
ENV DEBIAN_FRONTEND=noninteractive
RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
       python3 python3-numpy xvfb xauth x11-utils mesa-utils fonts-liberation \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://dl.winehq.org/wine-builds/winehq.key -o /etc/apt/keyrings/winehq-archive.key \
    && curl -fsSL https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources \
       -o /etc/apt/sources.list.d/winehq-noble.sources \
    && apt-get update \
    && apt-get install -y --install-recommends winehq-stable=${WINE_VERSION} \
       libgl1-mesa-dri libglx-mesa0 libgl1-mesa-dri:i386 libglx-mesa0:i386 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe \
    -o /opt/python-installer.exe
RUN groupadd --gid 10001 worker \
    && useradd --uid 10001 --gid 10001 --create-home worker \
    && mkdir -p /work /sessions && chown 10001:10001 /work /sessions
COPY docker/*.sh docker/*.py tools/bench.py tools/profile_game.py tools/winproc.py /opt/worker/
COPY --from=wheels /wheels/environment/ /opt/wheels/environment/
# The same package for the Linux side: a trainer's VectorSession drives Wine workers from here.
COPY src/wc3env/ /opt/host/wc3env/
ENV PATH=/opt/wine/bin:$PATH WINEFSYNC=1 \
    DISPLAY=:99 WINEPREFIX=/home/worker/.wine WINEARCH=wow64 \
    WINEDLLOVERRIDES="mscoree,mshtml=" WINEDEBUG=-all \
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1
USER 10001:10001
WORKDIR /work
# Wine runs 32-bit code, which some image builders cannot (Modal's): INIT_WINE=0 leaves the prefix
# for the first container to initialize with install.sh (docker/modal_run.py snapshots it).
ARG INIT_WINE=1
RUN if [ "$INIT_WINE" = 1 ]; then bash /opt/worker/with-display.sh bash /opt/worker/install.sh; fi
# The initialized prefix lives in each container's private writable layer.
COPY --chown=10001:10001 assets/game/ /opt/game/
ENTRYPOINT ["bash", "/opt/worker/with-display.sh", "bash", "/opt/worker/entrypoint.sh"]
CMD ["smoke"]

# Optional agent layer; wc3env itself does not depend on wc3agent.
FROM environment AS agent
COPY --from=wheels /wheels/agent/ /opt/wheels/agent/
ARG INIT_WINE=1
RUN if [ "$INIT_WINE" = 1 ]; then bash /opt/worker/with-display.sh bash /opt/worker/install.sh agent; fi

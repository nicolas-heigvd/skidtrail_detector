ARG DEBIAN_FRONTEND=noninteractive

#FROM python:3.9.23-slim-trixie
FROM python:3.9-slim AS build

ARG DEBIAN_FRONTEND

LABEL org.opencontainers.image.authors="Nicolas Blanc @ HEIG-VD"
LABEL org.opencontainers.image.title="Skid Trail Detector"
LABEL org.opencontainers.image.description="A detector skid trails in forests. Based on the work from RaffBienz."
LABEL org.opencontainers.image.source="https://github.com/nicolas-heigvd/"
LABEL org.opencontainers.image.url="https://github.com/nicolas-heigvd/"
LABEL org.opencontainers.image.documentation="https://github.com/nicolas-heigvd/.../blob/main/README.md"
LABEL org.opencontainers.image.licenses="GPL-3.0-only"
LABEL org.opencontainers.image.vendor="HES-SO/HEIG-VD"
LABEL org.opencontainers.image.created="2025-11-20T20:00:00Z"
LABEL license-url="https://www.gnu.org/licenses/gpl-3.0.html"

# Create a new group and a new user with your UID/GID
RUN groupadd -g 1000 skid
RUN useradd -u 1000 -g 1000 -m skid

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
RUN apt-get -yq update \
    && apt-get install -yq \
    --fix-missing \
    --no-install-recommends \
    libexpat1 \
    && apt-get -yq autoremove --purge \
    && apt-get -yq autoclean \
    && ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

COPY requirements/* ./requirements/

RUN python -m pip install --trusted-host pypi.python.org pip==25.2 \
  && pip install --trusted-host pypi.python.org --upgrade pip-tools \
  && pip install --trusted-host pypi.python.org -r ./requirements/requirements.txt

COPY src /app/

# Stage 2: Production Stage
FROM build AS prod

WORKDIR /app

# Copy the production requirements
COPY --from=build /app/requirements/requirements.txt ./requirements/
RUN pip install --trusted-host pypi.python.org -r requirements/requirements.txt

# Switch to this user
USER skid
ENTRYPOINT ["python3"]
#CMD ["main.py"]

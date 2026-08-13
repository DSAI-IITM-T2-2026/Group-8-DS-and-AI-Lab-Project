# Build context is "Milestone 6/Application/frontend".
#
#   docker build -t wildfire-frontend .
#
FROM node:20-slim AS build
WORKDIR /app

COPY package.json package-lock.json* .npmrc* ./
RUN npm ci

COPY . .
# Produces dist/client/index.html. The build script also writes
# dist/server/index.js and dist/.openai/hosting.json for an unrelated
# OpenAI "Sites" hosting target; only dist/client is used below.
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist/client /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

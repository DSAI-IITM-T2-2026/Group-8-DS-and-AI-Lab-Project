/**
 * California coverage explorer — theme-aware, selector-driven (D3).
 * Fits explicitly to CA lon/lat (never default world mercator).
 * Base shapes once; month changes only mutate classes / opacity / strip.
 */
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

/** Simplified California coastline (lon, lat) — GeoJSON order */
const CA_RING = [
  [-124.41, 42.01], [-124.0, 41.0], [-124.2, 40.0], [-123.7, 39.0],
  [-123.5, 38.2], [-122.5, 37.8], [-122.4, 37.1], [-121.9, 36.3],
  [-121.5, 35.5], [-120.6, 34.6], [-119.8, 34.4], [-119.0, 34.1],
  [-118.5, 33.8], [-117.3, 32.7], [-116.5, 32.6], [-115.0, 32.7],
  [-114.6, 33.0], [-114.5, 34.0], [-114.6, 34.9], [-114.8, 35.8],
  [-116.0, 36.5], [-117.0, 37.2], [-118.0, 37.8], [-119.0, 38.5],
  [-120.0, 39.0], [-120.0, 42.0], [-124.41, 42.01],
];

const LON0 = -124.7;
const LON1 = -113.9;
const LAT0 = 32.3;
const LAT1 = 42.3;

const EXPECTED_S2 = 6;
const EXPECTED_S5P = 28;

const root = document.getElementById("viz-root");
const selectEl = document.getElementById("selector");
const summaryEl = document.getElementById("summary");
const metaEl = document.getElementById("meta");
const stripEl = document.getElementById("windowStrip");
const svgEl = document.getElementById("map");

let data = null;
let layers = null;

function boxPoly(b) {
  if (!b || b.west == null || b.east == null) return null;
  const w = +b.west;
  const e = +b.east;
  const s = +b.south;
  const n = +b.north;
  if (![w, e, s, n].every(Number.isFinite)) return null;
  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [[[w, n], [e, n], [e, s], [w, s], [w, n]]],
    },
  };
}

function caFeature() {
  return {
    type: "Feature",
    geometry: { type: "Polygon", coordinates: [CA_RING] },
  };
}

/** Equirectangular fit to fixed CA frame — reliable, no world-scale collapse */
function makeProjection(width, height, pad = 28) {
  const x0 = pad;
  const x1 = width - pad;
  const y0 = pad;
  const y1 = height - pad - 18;
  return d3
    .geoTransform({
      point(lon, lat) {
        const x = x0 + ((lon - LON0) / (LON1 - LON0)) * (x1 - x0);
        const y = y0 + ((LAT1 - lat) / (LAT1 - LAT0)) * (y1 - y0);
        this.stream.point(x, y);
      },
    });
}

function monthKeys(payload) {
  const set = new Set();
  (payload.windows || []).forEach((k) => set.add(k));
  Object.keys(payload.s2_windows_by_month || {}).forEach((k) => set.add(k));
  Object.keys(payload.s5p_windows_by_month || {}).forEach((k) => set.add(k));
  return Array.from(set).sort();
}

function coverageFor(monthKey) {
  const s2 = (data.s2_windows_by_month && data.s2_windows_by_month[monthKey]) || [];
  const s5p = (data.s5p_windows_by_month && data.s5p_windows_by_month[monthKey]) || [];
  const s2Present = s2.filter((w) => w.present !== false).length;
  const s5pPresent = s5p.filter((w) => w.present !== false).length;
  return {
    s2,
    s5p,
    s2Present,
    s5pPresent,
    s2Score: Math.min(1, s2Present / EXPECTED_S2),
    s5pScore: Math.min(1, s5pPresent / EXPECTED_S5P),
  };
}

function pickDefaultMonth(months, payload) {
  let best = null;
  let bestScore = -1;
  for (const m of months) {
    const s2 = (payload.s2_windows_by_month?.[m] || []).length;
    const s5p = (payload.s5p_windows_by_month?.[m] || []).length;
    const score = s2 * 10 + s5p; // prefer months with S2
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  return best || months[0] || "";
}

function describeSelection(monthKey, cov) {
  if (cov.s2Present === 0 && cov.s5pPresent === 0) {
    return `${monthKey}: no S2 or S5P CSV windows on GCS for this month.`;
  }
  const parts = [];
  parts.push(cov.s2Present ? `S2 ${cov.s2Present}× ~5-day windows` : "S2 absent");
  parts.push(cov.s5pPresent ? `S5P ${cov.s5pPresent} daily windows` : "S5P absent");
  const nrows = (data.csv_grid_extent && data.csv_grid_extent.nrows) || "413k";
  return `${monthKey}: ${parts.join(" · ")} · CSV grid ~${nrows} cells.`;
}

function chipRow(ids, cls) {
  if (!ids.length) return `<span class="chip empty">none</span>`;
  return ids
    .slice()
    .sort((a, b) => a - b)
    .map(
      (id) =>
        `<span class="chip ${cls}" title="window ${String(id).padStart(3, "0")}">${String(id).padStart(3, "0")}</span>`
    )
    .join("");
}

function renderStrip(cov) {
  const s2Ids = cov.s2.map((w) => Number(w.window));
  const s5pIds = cov.s5p.map((w) => Number(w.window));
  stripEl.innerHTML = `
    <div class="strip-block">
      <h3>S2 windows (${cov.s2Present})</h3>
      <div class="chips">${chipRow(s2Ids, "on-s2")}</div>
    </div>
    <div class="strip-block">
      <h3>S5P daily windows (${cov.s5pPresent})</h3>
      <div class="chips">${chipRow(s5pIds, "on-s5p")}</div>
    </div>
  `;
}

function buildMap() {
  const width = Math.max(320, root.clientWidth || 736);
  const height = Math.round(width * 0.85);

  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();
  svg
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  const projection = makeProjection(width, height);
  const path = d3.geoPath(projection);

  const ca = caFeature();
  const extent = boxPoly(data.csv_grid_extent || data.intended_aoi || data.aoi_bounds);
  const aoi = boxPoly(data.aoi_bounds);
  const intended = boxPoly(data.intended_aoi);

  const g = svg.append("g").attr("class", "map-layers");

  // Land wash behind everything
  g.append("path")
    .datum(ca)
    .attr("class", "ca-fill")
    .attr("d", path);

  const extentPath = extent
    ? g.append("path").datum(extent).attr("class", "extent-fill is-absent").attr("d", path)
    : null;

  // S5P secondary ring (mutated via opacity class)
  const s5pRing = extent
    ? g.append("path").datum(extent).attr("class", "s5p-ring is-off").attr("d", path)
    : null;

  if (intended) {
    g.append("path").datum(intended).attr("class", "intended-box").attr("d", path);
  }
  if (aoi) {
    g.append("path").datum(aoi).attr("class", "aoi-box").attr("d", path);
  }

  g.append("path").datum(ca).attr("class", "ca-outline").attr("d", path);

  // Status pills drawn in SVG (always readable)
  const statusG = svg.append("g").attr("class", "status-pills").attr("transform", `translate(20, ${height - 36})`);
  statusG
    .append("text")
    .attr("class", "label-text")
    .attr("id", "mapStatusText")
    .attr("y", 0)
    .text("Select a month");

  layers = { extentPath, s5pRing, width, height };
}

function update(monthKey) {
  if (!data || !layers) return;
  const cov = coverageFor(monthKey);

  if (layers.extentPath) {
    layers.extentPath
      .classed("is-absent", cov.s2Present === 0)
      .classed("is-partial", cov.s2Present > 0 && cov.s2Score < 0.99)
      .classed("is-complete", cov.s2Score >= 0.99)
      .style("fill-opacity", cov.s2Present === 0 ? 0.45 : 0.22 + 0.58 * cov.s2Score);
  }

  if (layers.s5pRing) {
    layers.s5pRing
      .classed("is-off", cov.s5pPresent === 0)
      .classed("is-on", cov.s5pPresent > 0)
      .style("stroke-opacity", cov.s5pPresent === 0 ? 0 : 0.35 + 0.55 * cov.s5pScore);
  }

  const status = d3.select("#mapStatusText");
  if (!status.empty()) {
    status.text(
      `S2 ${cov.s2Present}/${EXPECTED_S2} windows · S5P ${cov.s5pPresent} days`
    );
  }

  renderStrip(cov);
  summaryEl.textContent = describeSelection(monthKey, cov);

  const aoi = data.aoi_bounds || {};
  const be = data.binding_edges || {};
  metaEl.textContent = [
    data.verdict ? String(data.verdict).split(".")[0] + "." : "",
    `AOI W ${aoi.west} · S ${aoi.south} · E ${aoi.east} · N ${aoi.north}`,
    be.west
      ? `Binding: ${["west", "south", "east", "north"].map((k) => `${k}=${be[k]}`).join(", ")}`
      : "",
  ]
    .filter(Boolean)
    .join("  ·  ");
}

function init(payload) {
  data = payload;
  const months = monthKeys(payload);
  selectEl.innerHTML = months.map((m) => `<option value="${m}">${m}</option>`).join("");
  selectEl.value = pickDefaultMonth(months, payload);

  buildMap();
  selectEl.addEventListener("change", () => update(selectEl.value));

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      buildMap();
      update(selectEl.value);
    }, 120);
  });

  update(selectEl.value);
}

fetch("./coverage_map_data.json")
  .then((r) => {
    if (!r.ok) throw new Error("coverage_map_data.json missing — run scripts/verify_gcs_data.py");
    return r.json();
  })
  .then(init)
  .catch((err) => {
    summaryEl.textContent =
      "Could not load coverage_map_data.json. Run: python scripts/verify_gcs_data.py --year 2025";
    metaEl.textContent = String(err);
  });

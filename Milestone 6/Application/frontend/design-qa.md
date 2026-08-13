# Prediction Studio design QA

## Comparison target

- Source visual truth: `reference/prediction-studio-approved.png`
- Final desktop implementation: `qa/implementation-final.png`
- Final mobile implementation: `qa/implementation-mobile-final.png`
- Normalized full-view comparison: `qa/comparison-final.png`
- Focused comparisons: `qa/comparison-map-final.png`, `qa/comparison-top-final.png`, and `qa/comparison-table-final.png`
- Source and desktop implementation: 1672 × 941 pixels
- CSS viewport: 1672 × 941 at device scale factor 1
- Density normalization: none required; the source and desktop implementation were captured at identical pixel dimensions before being scaled equally in the comparison frame.
- Verified state: Live tab, Tuolumne County selected, May 18 at 10:00 AM, baseline inputs, prediction loaded, validation page one, and development adapters enabled.

## Final comparison

The implementation preserves the reference's major frame and hierarchy: dark navigation rail, two-tier workspace header, map-dominant left column, scenario and prediction cards, three explanation cards, and a dense validation table. Panel borders, radii, orange active states, chart semantics, table badges, and vertical rhythm are aligned closely with the supplied reference.

Focused evidence confirms:

- the county layer remains geographically registered while the MapLibre basemap pans and zooms;
- the five feature rows, gauges, and SHAP labels remain visible without collisions;
- validation columns, filters, status badges, result count, and pagination are preserved at the reference density;
- the narrow layout has no document-level horizontal overflow and exposes navigation through the mobile menu.

Intentional product-safe differences:

- The UI identifies development data and mock provenance instead of presenting fixture results as production inference.
- An explicit `Run Scenario` action prevents accidental inference on every slider movement.
- The live OpenStreetMap basemap has different label styling from the reference image, while the county geometry remains interactive and correctly registered.
- Slider bounds use centralized feature metadata (`DEM` 0–2,790 and `ERF5` 0–1.00) so the production model contract can replace the adapter without a UI rewrite.

## Comparison history

### Iteration 1 — blocked

- [P1] The county overlay and basemap used separate coordinate systems and drifted during pan or zoom.
- [P1] Every feature input exposed overlapping `progress` and `range` controls; the range hit area and keyboard behavior were unreliable.
- [P1] Validation could reload continuously after page one and could render empty pages while claiming 25 results.
- [P1] Narrow layouts clipped navigation, scenario values, units, and table controls.
- [P2] Menus were not dismissible, several visible controls were inert, and dense typography was inconsistent.

Fixes: projected county paths from MapLibre coordinates on every map movement; replaced the stacked controls with one reusable semantic slider; corrected validation request dependencies and supplied a truthful 25-row fixture set; moved filters before pagination; implemented mobile navigation and responsive control layouts; and added outside-click/Escape handling to menus.

### Iteration 2 — blocked

- [P2] At 1280 × 800, explanation cards collided with the validation region.
- [P2] Mobile validation actions still competed for width.
- [P2] Recharts skipped or wrapped feature labels and emitted container-size warnings during compact renders.

Fixes: added compact-height chart and table rules; stacked mobile validation controls; forced all categorical ticks to render with a custom label; and supplied deterministic initial chart dimensions.

### Iteration 3 — passed

- No actionable P0, P1, or P2 visual, behavioral, or responsive findings remain.
- Remaining P3 variation is limited to live basemap tile styling and future production feature metadata.

## Interaction and runtime evidence

Browser-tested flows:

- selected a county directly on the map, zoomed, reset the view, and changed risk filters;
- changed a feature with keyboard arrows and confirmed that only one slider semantic is exposed per feature;
- changed the timeline date and exercised play/pause;
- loaded a preset, ran a simulated scenario, and reset to baseline;
- opened and dismissed Help, Saved Scenarios, filter, date, profile, and preset menus;
- opened placeholder navigation, returned to Prediction Studio, and used collapse/expand plus the mobile menu;
- paged to the fifth validation page and applied service-backed date and region filters;
- checked a clean final desktop reload and a final mobile reload: zero new browser warnings or errors.

Automated verification:

- TypeScript typecheck: passed
- Vitest: 11 tests passed
- Sites worker tests: 4 tests passed
- Production build: passed

## Required fidelity surfaces

- Typography: self-hosted Inter, consistent weights and compact data hierarchy, with readable labels at desktop and mobile sizes.
- Spacing: reference-aligned panel grid and padding, with responsive reflow instead of clipped desktop tracks.
- Color: orange probability language, green result semantics, red/blue SHAP direction, and matching dark navigation.
- Imagery and visualization: live vector county geometry and charts; no stretched screenshot substitutes.
- Copy: approved screen labels with explicit development-data disclosures where production claims are not yet verified.
- Accessibility: semantic tabs and buttons, one range input per feature, visible focus states, keyboard-operable menus, `aria-current` pagination, and accessible chart summaries.

final result: passed

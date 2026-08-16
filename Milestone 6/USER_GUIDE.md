# Wildfire IQ User Guide

Wildfire IQ creates a next-day wildfire risk forecast for supported California
grid cells. It prepares the required data, runs the model, and shows the cells
that need the most attention.

> Wildfire IQ is a decision-support tool, not an emergency warning system.
> Follow official local and state guidance during a wildfire emergency.

## Access the application

Open [Wildfire IQ](http://34.9.154.237:8080/) in your web browser.

## Create a forecast

1. Open the application using the link above.
2. Under **Select prediction date**, use the default **Tomorrow** selection or
   choose an earlier date within the displayed range.
3. Select **Generate wildfire forecast**.
4. Follow the progress shown under **Forecast run**.
5. When processing finishes, review the map and ranked results.

The app may reuse data that is already available or prepare missing weather,
fire, and satellite inputs. Satellite exports can take 30 minutes or longer.
You may close the page while waiting; the worker continues in the background.

While a forecast is queued, running, or waiting for external data, select
**Stop forecast** beside its status to stop local preparation. The run changes
to **Stopped** and can be started again later. Cloud exports already submitted
before stopping may continue remotely; their completed data is reused by the
next run rather than discarded.

Tomorrow is handled conservatively. The app uses an existing validated daily
artifact or proceeds only when every required causal source file is already
available. It does not launch long cloud preparation jobs for tomorrow. If the
source set is incomplete, the page simply reports that tomorrow's data is not
available yet; check again later.

## Understand the model evaluation scorecard

The **Champion model evaluation** scorecard summarizes performance on the
held-out 2025 test set: 93,518 cell-days with 1,325 positive cell-days. It
reports PR-AUC, ROC-AUC, Recall@25, Precision@25, Brier score, and PR-AUC lift
over the Milestone 5 naive baseline.

These are historical model-level evaluation results. They are not measured
performance for tomorrow or for another date selected in the app.

## Read the results

- **Daily priority map:** darker cells have higher priority for the selected day.
- **Highest-ranked cells:** cells are ordered from highest to lowest priority.
- **Top-25 alert:** the cell is among the day's 25 highest-priority cells. This
  is a model alert, not an official public-safety alert.
- **Calibrated probability:** the model's estimated fire probability for a cell.
- **Priority score:** a daily comparison score used to rank cells.

Select a cell on the map or in the ranked list to see its probability, rank,
alert state, model version, and strongest model drivers when available.

## Compare a completed forecast with FIRMS observations

For a completed historical day, select **Actual vs Top-25** above the map. The
map then shows cells that were observed and captured, observed but missed,
alerted without an observation, or neither. Patterns and labels accompany the
colors. Five cards summarize FIRMS-observed cells, captures, Recall@25,
Precision@25, and false alerts. Select a cell to inspect its FIRMS pixel count,
maximum confidence, capture outcome, and label source.

These observations are FIRMS thermal detections mapped to model cells, not
official wildfire perimeters, acreage, or incident confirmations. Labels are
available only after the California day ends and the completed FIRMS data
arrives. For today or tomorrow, the forecast map remains usable and the app
shows **Observed labels not available yet**. Recall displays `—` on a day with
no observed fire cells.

## If something goes wrong

- **Forecast is still waiting:** Earth Engine may still be preparing satellite
  data. Leave the run active and check again later.
- **Tomorrow's data is not available yet:** no cloud preparation was started.
  Check again after the daily source files have arrived.
- **Preparation needs attention:** read the message and select **Retry**.
- **Scoring needs attention:** select **Retry scoring**.
- **The app does not load:** contact the application administrator and include
  the selected date and the error message shown on screen.

## Good to know

- Forecasts use only information available before the selected prediction day.
- Map areas are model grid cells, not counties.
- A high rank is relative to other cells on that day; it does not guarantee
  that a wildfire will occur.

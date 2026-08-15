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
2. Under **Select prediction date**, choose a date within the displayed range.
3. Select **Generate wildfire forecast**.
4. Follow the progress shown under **Forecast run**.
5. When processing finishes, review the map and ranked results.

The app may reuse data that is already available or prepare missing weather,
fire, and satellite inputs. Satellite exports can take 30 minutes or longer.
You may close the page while waiting; the worker continues in the background.

## Read the results

- **Daily priority map:** darker cells have higher priority for the selected day.
- **Highest-ranked cells:** cells are ordered from highest to lowest priority.
- **Top-25 alert:** the cell is among the day's 25 highest-priority cells. This
  is a model alert, not an official public-safety alert.
- **Calibrated probability:** the model's estimated fire probability for a cell.
- **Priority score:** a daily comparison score used to rank cells.

Select a cell on the map or in the ranked list to see its probability, rank,
alert state, model version, and strongest model drivers when available.

## If something goes wrong

- **Forecast is still waiting:** Earth Engine may still be preparing satellite
  data. Leave the run active and check again later.
- **Preparation needs attention:** read the message and select **Retry**.
- **Scoring needs attention:** select **Retry scoring**.
- **The app does not load:** contact the application administrator and include
  the selected date and the error message shown on screen.

## Good to know

- Forecasts use only information available before the selected prediction day.
- Map areas are model grid cells, not counties.
- A high rank is relative to other cells on that day; it does not guarantee
  that a wildfire will occur.

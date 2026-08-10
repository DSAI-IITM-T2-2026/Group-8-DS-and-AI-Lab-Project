# AI-Powered Wildfire Early Detection System
## Non-Technical Report

---

## Executive Summary

Wildfires represent one of the most destructive natural hazards facing California and the western United States, claiming lives, destroying homes, and damaging ecosystems every year. Traditional wildfire response is often **reactive**-firefighters dispatch only *after* smoke is spotted or flames have already grown into uncontrollable blazes.

Our project introduces an **AI-Powered Wildfire Early Detection System**. By continuously analyzing weather physics, landscape topography, vegetation dryness, and satellite imagery, our system calculates daily wildfire hazard levels across California. 

Instead of overwhelming emergency crews with thousands of alerts, this project identifies the **Top 25 Highest-Risk Zones Every Day**, giving organizations like CAL FIRE a crucial one day head start to pre-position firefighting crews, pre-stage water tankers, and protect vulnerable communities before ignitions occur.

---

## The Challenge: Why Early Detection Matters

In recent years, rising global temperatures, prolonged droughts, and severe seasonal winds have made California's wildfire seasons longer and more dangerous. 

### The Core Problems:
1. **The Speed of Wildfires**: Under high winds and bone-dry air, a small spark can transform into a roaring wildfire in minutes.
2. **Limited Response Resources**: Emergency services have a finite budget of firefighters, reconnaissance aircraft, and ground vehicles on any given day. They cannot patrol every mile of the state simultaneously.
3. **Late Notice**: Waiting for human emergency calls or satellite smoke detection often means responding too late to prevent catastrophic loss.

To solve this, emergency management needs a **predictive early-warning system** that acts like a 24/7 digital weather and landscape observer.

---

## How the Solution Works (In Simple Terms)

Think of our AI system as an expert digital analyst constantly reading the "pulse" of California's environment every morning.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DAILY ENVIRONMENTAL INPUTS                      │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ Weather Physics   │ Terrain Shape     │ Satellite Context              │
│ • Air Temperature │ • Mountain Slopes │ • Plant & Fuel Dryness         │
│ • Air Dryness     │ • Elevation       │ • Air Pollution (Carbon Mon.)  │
│ • Wind Velocity   │ • Solar Exposure  │ • Neighboring Active Fires     │
└───────────────────┴───────────────────┴────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        ARTIFICIAL INTELLIGENCE                         │
│   Evaluates 86 physical & spatial indicators across 437 regional zones │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    DAILY TOP-25 EMERGENCY ALERT ROSTER                 │
│   Ranks California's highest-risk zones to guide daily fire patrols    │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. What the System Observes:
* **Atmospheric Drying Power (Vapor Pressure Deficit)**: Measures how aggressively the warm air is pulling moisture out of trees and living plants.
* **Wind Dynamics**: Tracks sustained winds and peak wind gusts that can carry sparks across valleys.
* **Fuel Moisture Deficits**: Combines heat exposure with soil moisture levels to identify vegetation that has reached explosive dryness.
* **Regional Fire History**: Tracks whether active fires are burning in neighboring areas and which direction prevailing winds are pushing them.

### 2. How the AI Makes Decisions:
The system breaks California down into a structured grid of **437 high-risk regional zones**. Every morning, the AI processes 86 environmental signals for each zone and assigns a calibrated **Wildfire Hazard Score**.

---

## The Top-25 Alert Strategy: Smart Resource Allocation

A critical innovation of this project is the **Top-25 Daily Alert Strategy**. 

If an AI flags 200 locations as "dangerous" every summer day, firefighters suffer from alert fatigue and cannot act on the information. 

By using advanced ranking algorithms, our AI compares all 437 regions against one another *on that specific calendar day* and highlights the **exact 25 most critical zones**. This mirrors how real-world fire agencies allocate daily reconnaissance flights and staging crews.

```
       [437 California Regional Zones Evaluated Every Morning]
                                 │
                                 ▼
         [AI Ranks All Regions from Highest to Lowest Risk]
                                 │
                                 ▼
     TOP 25 HIGHEST-RISK ZONES   →  Pre-position Fire Crews & Tankers
   412 SAFE / LOWER-RISK ZONES   →  Standard Monitoring
```

---

## Real-World Impact & Results

Our system was evaluated on a completely held-out benchmark dataset covering real California weather and fire events:

* **Over 40% Daily Fire Detection**: By monitoring just the Top 25 alerted zones each day, the system captures **over 41% of all wildfires** statewide.
* **Exceptional Performance During High Risk**: During extreme weather events (such as severe heatwaves or high wind gusts above 35 km/h), the system’s accuracy jumps-capturing **up to 68% of all wildfires**.
* **Zero False Alarms in Non-Burnable Areas**: The system automatically filters out urban centers and barren deserts, ensuring 100% of daily alerts are focused on burnable forests and brushlands.
* **Fast & Lightweight**: The entire statewide analysis finishes in **less than 15 milliseconds** on a standard laptop computer, making it suitable for field deployment in command vehicles.

---

## Public Access & Interactive Dashboard

To make these predictions accessible to firefighters, city planners, and the general public, the project includes an **interactive web dashboard**:

1. **Interactive Statewide Risk Map**: Displays a visual color-coded map of California showing safe zones (green), alert zones (red), and predicted danger hot-spots.
2. **Daily Top-25 Roster Table**: Provides a clean, sorted list of the day's highest-risk zones with detailed weather metrics.
3. **Transparent Danger Indicators**: Explains *why* a region is at risk (e.g., "High Wind Gusts + Extreme Vegetation Dryness").

---


## Project Team & Acknowledgments

This project was developed by **Group 8** as part of the DS & AI Lab Project:

| Member              | Roll Number | Signature Commit |
| ------------------- | ----------- | ---------------- |
| Ripunjay Kumar      | 21F3002511  |                  |
| Lakshay Garg        | 21F3001076  |                  |
| Roushan Kumar Singh | 23F1002240  |                  |
| Lakshmi Sruthi K    | 21F1005626  |                  |
| R Aditya            | 21F1004839  |                  |


# AI Powered Wildfire Early Detection and Alerting System using Multi Source Remote Sensing 
Wildfires are a recurring and destructive environmental disaster affecting forest ecosystems, biodiversity, human settlements, and infrastructure across the world. In many cases, wildfire events occur under predictable environmental conditions, such as prolonged low rainfall, high temperature, low humidity, and dry vegetation.

An official incident report by the Ministry of Home Affairs: Disaster Management Division [(Link to the Report)](https://www.ndmindia.mha.gov.in/ndmi/viewUploadedDocument?uid=D112) during the April 2021 Uttarakhand forest fire incident highlighted that significantly below-normal rainfall and low humidity were key contributing factors to widespread fire occurrences across multiple districts.

Despite such identifiable precursor conditions, current wildfire management systems are largely reactive, relying on:
-   Satellite-based hotspot detection (post-ignition)
-   Ground reporting after fire outbreak
-   Emergency response after fire spread
    

This results in delayed intervention and reduced ability to prevent large-scale damage.

### Core Problem:
There is a lack of an intelligent system that can analyze spatio-temporal satellite imagery and environmental conditions to predict wildfire risk before ignition, enabling early warning and proactive disaster mitigation.


## Our Plan
The proposed solution is an AI-based multimodal spatio-temporal wildfire early warning system that integrates satellite imagery, environmental data, and historical disaster records to predict wildfire risk before ignition.

### Approach:
The system uses three main data components:
-   Satellite imagery (spatial-temporal input):  Captures vegetation conditions, land cover, and changes in forest regions over time.
-   Environmental data (temporal input):  Includes temperature, humidity, wind speed, and rainfall, representing atmospheric conditions influencing fire risk.

## Who does it affect?

The impact of wildfire prediction and early warning systems spans multiple stakeholders:

-   Forest and wildlife protection departments
-   Disaster management authorities
-   Emergency response agencies (fire services, National Disaster Response Force etc.)
-   Communities living in or near forested and wildfire-prone regions 
-   Environmental conservation organizations
-   Wildlife protection agencies
-   Government policy makers and climate monitoring bodies

## What our Limitations Might Be

The proposed system has the following limitations:
-   The model predicts probabilistic wildfire risk, not exact ignition time or cause
-   Geographic scope is limited by data availability and resolution of satellite imagery
-   Human-induced ignition factors (intentional fires or accidental causes) are not directly observable in the data
-   Historical disaster reports are used only for labeling and validation, not as direct model inputs

## Our Core Objectives

- Develop a multimodal dataset integrating satellite imagery, environmental data, and historical disaster records
- Design a spatio-temporal deep learning model for wildfire risk prediction
- Predict wildfire risk 24 to 48 hours before potential ignition events
- Generate geospatial wildfire risk heatmaps for forest regions
- Implement an automated early warning alert system for high risk areas

**Model Level Evaluation Metrics**: 
- Primary: Accuracy and Recall Primarily (Better Safe than Sorry)
- Secondary: Precision and F1-score


## Literature Review & Existing Solutions

Wildfire detection research has shifted from simple **threshold-based thermal detection** toward **deep learning models** built on multispectral satellite imagery. Most studies draw on freely available data from *Sentinel-2, Landsat-8/9, MODIS, VIIRS,* and *NASA FIRMS*, using architectures such as CNNs, U-Net, DeepLabV3+, Vision Transformers, and attention-based models.

Across the literature, a clear pattern emerges: deep learning models now regularly achieve **90%+ accuracy**, and some, like *MobileNetV2* on Sentinel-2 imagery, reach near-perfect F1 scores with fast inference. Operationally, *NASA FIRMS* remains the backbone of real-world alerting, using MODIS/VIIRS thermal hotspots to deliver near real-time detection. However, nearly every study shares the same shortcomings — reliance on a **single image or single region**, **no temporal modeling**, and detection that only happens *after* a fire has already ignited. Commercial and institutional platforms like Copernicus EMS, Google Earth Engine, and OroraTech extend this further with mapping and processing power, but none combine prediction, multi-source fusion, and accessibility in one system.

| Source | Data Used | Method | Key Result | Main Limitation |
|---|---|---|---|---|
| Uni-temporal Sentinel-2 Study (2023) | Sentinel-2 | U-Net + ResNet50 | F1 98.78%, IoU 97.38% | Single image only |
| Forest Fire Surveillance Review (2024) | Multiple | CNN, YOLO, Transformer | Most studies >90% accuracy | No standard benchmark |
| CNN Comparison (2025) | Sentinel-2 | MobileNetV2 | F1 ≈ 0.99, fast inference | Single region evaluation |
| Copernicus EMS | Sentinel | Emergency mapping | Fast response mapping | Not built for prediction |


### Strengths, Weaknesses & Research Gaps

The field's **strengths** are well established: high detection accuracy from deep learning, global coverage via free satellite imagery, automated large-area surveillance, and mature, well-tested segmentation architectures.

Its **weaknesses**, however, define the research opportunity. Most methods use *only* optical or thermal imagery, rarely fusing it with weather, terrain, or atmospheric data. Temporal evolution is almost never modeled, cloud cover still degrades optical performance, explainability is minimal, and top-performing models are often computationally heavy.

These weaknesses translate directly into research gaps:

- **Single-source imagery** is the norm: fuse optical, thermal, atmospheric, weather, and terrain data.
- **Temporal context is rare**: apply ConvLSTM or Temporal Transformers.
- **Smoke and aerosol cues are rarely used**:  integrate Sentinel-5P.
- **Risk estimation is mostly absent**: derive a risk score from weather and terrain.
- **Explainability is minimal**: apply Grad-CAM and SHAP.
- **False alarm rates are high**: use attention-guided multimodal fusion.

## Our Proposed Framework

1. **Multi-modal fusion** of free Earth observation sources.
2. **Spatio-temporal learning**, replacing single frame prediction.
3. **Vegetation stress indices** fused with thermal anomalies.
4. **Attention mechanisms** to reduce false positives.
5. **Explainable AI** for emergency decision support.
6. A **Unified Pipeline** covering detection, risk scoring, and alert generation end to end.

Compared to existing work, this framework adds atmospheric pollutants, strengthens weather and terrain fusion, introduces temporal learning and risk prediction where these are currently rare, builds in explainability from the start, and delivers a complete end-to-end alert system rather than a partial one.

## Methodology: Baselines, Metrics & Benchmarks

**Baseline models** span both classical and deep learning approaches — *Random Forest, SVM,* and *XGBoost* on the classical side; *U-Net, DeepLabV3+, YOLOv8, MobileNetV2, ResNet50,* and *Vision Transformer* on the deep learning side.

**Evaluation** will use *recall and accuracy as primary metrics with precision, F1-score, IoU,* and *ROC-AUC* as secondary metrics. 

**Industry benchmarks** for comparison include *NASA FIRMS, MODIS Active Fire Product, VIIRS Active Fire Product, Copernicus EMS,* and *Google Earth Engine*.


## References

1. [Brief Report on Forest Fire incident in Uttarakhand as on 04th April, 2021 at 2000 Hrs](https://www.ndmindia.mha.gov.in/ndmi/viewUploadedDocument?uid=D112)
2. [Deep Learning Approaches for Wildland Fires Using Satellite Remote Sensing Data.](https://doi.org/10.3390/fire6050192)
3. [Deep Learning Approaches for Wildland Fires Remote Sensing: Classification, Detection and Segmentation.](https://doi.org/10.3390/rs15071821)
4. [Forest Fire Surveillance Systems: A Review of Deep Learning Methods.](https://doi.org/10.1016/j.heliyon.2023.e23127)
5. [Uni-temporal Sentinel-2 Imagery for Wildfire Detection Using Deep Learning Semantic Segmentation Models, 2023.](https://doi.org/10.1080/19475705.2023.2196370)
6. [Comparative Analysis of CNN Architectures for Satellite-Based Forest Fire Detection, 2025.](https://www.sciencedirect.com/science/article/abs/pii/S2352938525002927)
7. [Copernicus Sentinel Documentation.](https://emergency.copernicus.eu/)

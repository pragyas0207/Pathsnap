# Dataset Validation and Analysis

## 1. Introduction

Dataset validation is an important part of the PathSnap map-matching and route-reconstruction pipeline. Before performing feature engineering and model training, the input GPS trajectories and road-network data must be checked for structural consistency, missing values, invalid references, timestamp irregularities, and route connectivity.

This document summarises the validation and exploratory analysis performed on the training and student-testing datasets.

## 2. Dataset Overview

The dataset contains two main subsets:

- **Training dataset:** 80 trips
- **Student-testing dataset:** 10 trips

Each trip consists of four files:

| File | Description |
|---|---|
| `.track` | GPS observations containing longitude, latitude, and timestamp |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Directed connections between road-network nodes |
| `.route` | Reference route represented through arc identifiers |

The dataset contains the following records:

| Dataset | GPS observations | Road nodes | Road arcs | Route entries |
|---|---:|---:|---:|---:|
| Training | 208,417 | 1,933,485 | 3,876,904 | 31,336 |
| Student testing | 19,160 | 90,689 | 185,041 | 3,477 |

All 90 trips were checked for the presence of the four expected files.

## 3. Structural Validation

The first stage involved checking whether the input files followed the expected structure.

The following checks were performed:

1. Verified that every trip contained its corresponding `.track`, `.nodes`, `.arcs`, and `.route` files.
2. Checked the expected number of columns in each file.
3. Verified that numeric values could be interpreted correctly.
4. Checked for missing and non-finite numeric values.
5. Checked whether road-network references were within valid index ranges.

The expected column structures were:

- Track: 3 columns.
- Nodes: 2 columns.
- Arcs: 2 columns.
- Route: 1 column.

### Validation findings

- All 90 trips contained the expected four files.
- The observed column counts were consistent with the expected structures.
- No missing or non-finite numeric values were detected.
- All road-arc endpoint references were within the valid node index range.
- All reference-route arc identifiers were within the valid arc index range.

These checks establish that the dataset is structurally suitable for subsequent processing.

## 4. Timestamp and GPS Trajectory Analysis

GPS observations are associated with timestamps, which are important for understanding movement between consecutive observations.

The timestamps were checked for chronological consistency and sampling irregularities.

### Findings

- Timestamps were strictly increasing across the analysed trips.
- 60 out of 80 training trips had sampling intervals different from one second.
- 6 out of 10 student-testing trips had sampling intervals different from one second.
- 40 out of 80 training trips contained timestamp gaps greater than 10 seconds.
- 2 out of 10 student-testing trips contained timestamp gaps greater than 10 seconds.

The largest observed timestamp gaps were:

| Dataset | Largest timestamp gap |
|---|---:|
| Training | 153,064 seconds |
| Student testing | 210 seconds |

The unusually large training gap was observed in trip `00000090`.

### Why this matters

Irregular GPS sampling can affect movement-related calculations.

For example, calculating speed using the distance between two GPS observations requires the corresponding time difference:

**Speed = Distance / Time Difference**

Assuming a constant sampling interval would produce misleading movement estimates when observations are separated by different amounts of time.

Therefore, timestamp differences should be explicitly considered during feature engineering and sequential route reconstruction.

## 5. Road-Network Reference Validation

The road network is represented using nodes and directed arcs.

Each arc connects a starting node to an ending node. The reference route contains identifiers corresponding to rows in the arc file.

A route should ideally form a continuous sequence of connected arcs.

For two consecutive arcs, continuity is checked using:

`to_node(current_arc) == from_node(next_arc)`

If this condition is not satisfied, the two consecutive arcs are considered disconnected.

### Aggregate findings

Across the training dataset:

- 31,243 consecutive arc pairs were connected.
- 13 consecutive arc pairs were disconnected.
- Overall observed continuity was approximately 99.96%.
- 10 out of 80 training trips contained at least one disconnected consecutive arc pair.

### Example: Trip 00000001

For this trip:

- Number of reference-route arcs: 525.
- Number of consecutive arc pairs: 524.
- Connected consecutive pairs: 524.
- Disconnected consecutive pairs: 0.
- Route continuity: 100%.

### Why this matters

Route continuity is important because a sequence of individually valid road segments does not necessarily represent a valid connected path.

Checking connectivity helps identify inconsistencies in reference routes and provides useful information when evaluating map-matching predictions.

The small number of disconnected reference-route pairs should also be considered when interpreting route-level evaluation results.

## 6. Data Integrity Considerations

The validation process highlights several considerations for the downstream pipeline.

### 6.1 Irregular sampling intervals

GPS observations are not consistently separated by one second. Time-dependent features should therefore use actual timestamp differences.

### 6.2 Large observation gaps

Large timestamp gaps may indicate missing observations or interruptions in trajectory recording. Sequential decoding should account for these gaps rather than assuming continuous movement.

### 6.3 Reference-route inconsistencies

A small number of reference routes contain disconnected consecutive arcs. These cases should be identified during evaluation instead of assuming that every reference route is perfectly connected.

### 6.4 Geographic distance calculations

Longitude and latitude are angular coordinates. Direct Euclidean distance calculations on raw longitude and latitude values are not appropriate for accurate distance measurements.

Geographic coordinates should be transformed into a suitable projected coordinate system or evaluated using geodesic distance calculations when measuring distances in metres.

## 7. Implications for the PathSnap Pipeline

The findings from dataset validation are relevant to several stages of the PathSnap implementation.

| Pipeline stage | Relevant validation finding | Implication |
|---|---|---|
| Data loading | Consistent file structures | Input files can be processed using a common loading procedure |
| Geographic processing | Longitude and latitude coordinates | Use appropriate geographic projection for distance calculations |
| Feature engineering | Irregular timestamps | Calculate movement features using actual time differences |
| Candidate generation | Road-network node and arc references | Validate candidate identifiers against the road network |
| Sequential decoding | Large timestamp gaps | Account for gaps when evaluating transitions |
| Route evaluation | Disconnected reference arcs | Consider reference-route continuity when interpreting results |

## 8. Conclusion

The dataset validation process confirmed that all 90 analysed trips contain the expected input files and consistent file structures. No missing or non-finite numeric values were detected, and all node and arc references were within their valid index ranges.

The exploratory analysis also identified irregular GPS sampling intervals, large timestamp gaps, and a small number of disconnected consecutive reference-route arcs.

These observations are important for designing a reliable map-matching pipeline. In particular, they motivate the use of actual timestamp differences, appropriate geographic distance calculations, graph-connectivity checks, and careful interpretation of route-level evaluation metrics.

Dataset validation therefore provides an important foundation for the subsequent preprocessing, feature engineering, model training, and route reconstruction stages of PathSnap.
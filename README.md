# NEXUS: Adaptive Urban Routing for Mobility-Impaired Individuals via Graph Theory and Dynamic Impedance

## Abstract
Traditional routing systems optimize for distance or temporal efficiency, failing to account for the biomechanical cost imposed by topographic variance and micro-barriers (e.g., degraded surfaces, absence of sloped curbs). This project proposes NEXUS, a computational architecture that mitigates urban exclusion by calculating the route of minimal "biomechanical effort." Utilizing OpenStreetMap (OSM) data, Digital Elevation Models (DEM), and a Pessimistic Safe Fallback imputation heuristic, NEXUS applies an advanced dynamic routing algorithm (D* Lite) on a weighted spatial graph to dynamically recalibrate paths based on community-reported impediments (VGI).

## Project Architecture
* `data_pipeline/`: Handles ETL processes, spatial graph instantiation via OSMnx, and deterministic imputation of missing accessibility tags.
* `core/`: Contains the mathematical impedance model, dynamic graph state management, and the routing algorithms.
* `evaluation/`: Scripts for empirical validation, computing the Route Overlap Index, and generating spatial visualizations.

## Scientific Objectives
1. Extract and process geospatial, elevation, and infrastructure data.
2. Develop a modified cost function weighting micro-barriers and surface topology.
3. Implement a dynamic routing architecture capable of processing real-time community inputs without total graph recompilation.
4. Empirically validate the biomechanical efficiency of NEXUS against standard shortest-path algorithms.
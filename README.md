# Cumulative-travel-time-analysis
Python workflow to estimate cumulative driving time from the Wildlife Institute of India (WII), Dehradun, to CGHS healthcare facilities in the surrounding study area using a 50 m raster-based least-cost approach.  The analysis uses a road network with assigned or assumed driving speeds to generate a cumulative travel-time surface from WII. 


Input data
Edit the three input paths at the top of `wii_cghs_accessibility.py`:
Dehradun boundary polygon.
Road network line layer.
Hospital/clinic point layer.
The road layer must contain an `fclass` attribute. The `maxspeed` attribute
is optional. If a positive numeric speed is not available, the script uses
the class-based assumed speeds defined in the script.
All input layers must have a defined CRS. 
Run
```bash
python wii_cghs_accessibility.py
```
By default, the script uses a 50 m raster resolution. It geocodes WII using
OpenStreetMap, so an internet connection is required unless fixed coordinates
are enabled in the settings.
Method and limitations
The workflow rasterizes road speeds, identifies the rasterized road component
connected to the WII origin, and calculates least-cost cumulative travel time
along that component. Hospital/clinic points are clipped to the Dehradun
boundary.
This is a raster-based approximation, not turn-by-turn navigation. It does
not account for traffic, one-way restrictions, turn penalties, road closures,
vehicle restrictions, or grade-separated crossings. Assumed road speeds are
modelling assumptions, not observed speeds.
This version calculates road-only travel time. It does not create a
continuous off-road travel-time surface.

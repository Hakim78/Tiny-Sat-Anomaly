# Earth Textures - NASA Blue Marble

Download high-resolution Earth textures from NASA for realistic 3D visualization.

## Required Textures

| File | Description | Source |
|------|-------------|--------|
| `earth_daymap.jpg` | Day side Earth texture | NASA Blue Marble |
| `earth_clouds.png` | Cloud layer (transparent) | NASA Visible Earth |
| `earth_bump.jpg` | Elevation/topography map | NASA SRTM |
| `earth_specular.jpg` | Ocean reflectivity | NASA Blue Marble |

## Download Links

### Option 1: NASA Visible Earth (Recommended)
1. Visit: https://visibleearth.nasa.gov/collection/1484/blue-marble
2. Download "Blue Marble: Next Generation" (2004)
3. Rename files as indicated above

### Option 2: Solar System Scope (Easy)
1. Visit: https://www.solarsystemscope.com/textures/
2. Download Earth textures (8K available)
3. Rename to match expected filenames

### Option 3: Direct URLs (2K Resolution)
```bash
# Earth day map
curl -o earth_daymap.jpg "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"

# Earth clouds
curl -o earth_clouds.png "https://www.solarsystemscope.com/textures/download/2k_earth_clouds.jpg"

# Earth bump/elevation
curl -o earth_bump.jpg "https://www.solarsystemscope.com/textures/download/2k_earth_normal_map.jpg"

# Earth specular (ocean mask)
curl -o earth_specular.jpg "https://www.solarsystemscope.com/textures/download/2k_earth_specular_map.jpg"
```

## Recommended Resolutions

| Use Case | Resolution | Size |
|----------|------------|------|
| Development | 2K (2048x1024) | ~500KB each |
| Production | 4K (4096x2048) | ~2MB each |
| High Quality | 8K (8192x4096) | ~8MB each |

## Attribution

When using NASA imagery:
> "Image courtesy of NASA Goddard Space Flight Center"
> Source: NASA Visible Earth (https://visibleearth.nasa.gov)

## Notes

- The app works without textures (procedural fallback)
- Textures significantly improve visual quality
- Use compressed JPG for faster loading
- PNG only needed for cloud transparency

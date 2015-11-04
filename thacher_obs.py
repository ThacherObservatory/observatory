import ephem
thob = ephem.Observer()
thob.long = ephem.degrees("-119.1773417")
thob.lat = ephem.degrees("34.467028")
thob.elevation = 504.4 
thob.date = "2017/8/21" 
sun = ephem.Sun(thob)
print sun.alt, sun.az
print sun.ra, sun.dec

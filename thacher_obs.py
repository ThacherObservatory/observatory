import ephem
thob = ephem.Observer()
thob.long = ephem.degrees("-119.1773417")
thob.lat = ephem.degrees("34.467028")
thob.elevation = 504.4 
thob.date = "2010/1/1" 
jupiter = ephem.Jupiter(thob)
print jupiter.alt, jupiter.az
print jupiter.ra, jupiter.dec

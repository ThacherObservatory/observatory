"""
I need some help with the path manager thing
It's not recognizing weather and seeing
"""

import weather as w
import seeing as s

wpath = '/Users/nickedwards/python/data/weather/'
spath = '/Users/nickedwards/python/data/seeing/'
weather_data = w.get_data(dpath=wpath)
seeing_data = s.get_data(path=spath)

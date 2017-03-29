# -*- coding: utf-8 -*-
"""
Created on Sat Oct 03 17:01:06 2015

This is a script to copy the Cloud Sensor data from the default directory
to the DropBox.

This is needed because the Cloud Sensor software will delete the daily
data files every 5 days to save space. 
"""

import os,shutil,glob,time,datetime

outpath  = os.path.normpath('C:/Users/Weather/Dropbox (Thacher)/Observatory/CloudSensor/Data')
srcpath = os.path.normpath('C:/Users/Weather/Documents/ClarityII')

print 'Starting infinite loop to check for Cloud Sensor files...'
print ' '
while True:
    files = glob.glob(srcpath+'/20*-[0-9]*-[0-9]*.txt')

    now = datetime.datetime.now()

    print 'Checking for old files...'
    print ' '
    for file in files:

        tfile = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        if now > tfile + datetime.timedelta(days=1):    
            print 'Copying '+file+' to '+outpath
            shutil.copy2(file, outpath)
            shutil.copy(srcpath+'/longtermlog.txt',outpath)

    sleeptime = 86400
    print 'Sleeping for '+str(sleeptime)+' seconds'
    print ' '
    time.sleep(sleeptime)

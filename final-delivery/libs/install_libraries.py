# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:10:28 2020

@author: rubenandrebarreiro
"""

# Import Sub-Process Python's Library,
# as system
import sys as system

# Import Sub-Process Python's Library,
# as sub_process
import subprocess as sub_process


# Install required Python's Libraries
def install_required_system_libraries(installation_library):
    
    # Print information about the start of the Installation
    print( "Installing the Python's Library, with the pip command: '{}'\n".format(installation_library) )
    
    # Install 'kneed' Python's Library 
    sub_process.check_call([system.executable, '-m', 'pip', 'install', installation_library])
    
    # Print information about the start of the Installation
    print( "Installation finished!!!")
    print("The '{}' Python's Library, was successfully installed, with the pip command...\n\n".format(installation_library) )

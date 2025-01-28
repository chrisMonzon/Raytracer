# Raytracer
Raytracer built from scratch in Python that is capable of rendering spheres, planes, and triangles, provided by a txt file

# Running the program
After downloading the Makefile, in the command line, run: python raytracer.py $(textfile name)

# txt file syntax:

keywords:

png [width] [height] [output filename]

color [R value from 0.0 to 1.0] [G value] [B value]

sphere [x coordinate] [y coordinate] [z coordinate]

sun [ray's x direction] [y direction] [z direction]

plane [A] [B] [C] [D] 

xyz [vertex A position] [vertex B position] [vertex C position] 

tri [vertex index 1] [vertex index 2] [vertex index 3]

keyword notes:
- objects are visible in the coordinate range of [-1, 1] along the x, y, z axes
- keyword arguments should only be separated by whitespace
- color values set the color for subsequent spheres, suns, and triangles (given by the tri keyword)
- the keyword 'sun' creates an infinitely far away light source
- the A/B/C/D values of the plane satisify the equation Ax + By + Cz + D = 0)
- triangles are 1-indexed, so a sample triangle txt line would look like: 1 2 4

# mouse_ESN
# First Run controllerSRV.py on blender on Record mode
# This will record reservoir parameters to the file paramStatic and readouts and random outputs to paramWalking
# Second regress a wOut between readouts and random outputs by using fit_Wout.py which needs paramWalking1 file
# So rename the previous file to generate wOut
# The generated Wout will be written to wout file
# The controllerSRV file reads and uses wout in its Test mode
# controllerSRV.py also need paramStatic1 file for the reservoir parameters 

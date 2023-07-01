import matlab.engine

def matlab_on():

  """Creazione del Matlab engine 
  """
  !pip install matlabengine
  eng = matlab.engine.start_matlab()

#def matlab_off():

  """Matlab engine fermato
  """

 # eng.quit()

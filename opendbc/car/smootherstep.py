def smootherstep(p: float) -> float:
  """
  5th-order smootherstep in [0..1]
  6p^5 - 15p^4 + 10p^3
  """
  return 6*(p**5) - 15*(p**4) + 10*(p**3)

class SmootherstepTransition:
  def __init__(self, T: float = 0.25):
    self.T = T
    self.start_value = 0.0
    self.target_value = 0.0
    self.current_value = 0.0
    self.elapsed = 0.0  
        
  def set_target(self, new_target: float):
    self.start_value = self.current_value
    self.target_value = new_target
    self.elapsed = 0.0
        
  def update(self, dt: float) -> float:
    self.elapsed += dt
        
    if self.elapsed < self.T:
      p = self.elapsed / self.T
      alpha = smootherstep(p)
      self.current_value = (self.start_value + alpha*(self.target_value - self.start_value))
    else:
      self.current_value = self.target_value
        
    return self.current_value

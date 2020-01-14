#
# Forward kinematics of an 8 link robot arm -- 8nm = 8 inputs, high
# nonlinearity, med noise 
#
#
Origin: simulated

Usage: development

Order: uninformative

Attributes:
  1  theta1	      u  [-3.1416,3.1416]	# ang position of joint 1 in radians
  2  theta2	      u  [-3.1416,3.1416]	# ang position of joint 2 in radians
  3  theta3	      u  [-3.1416,3.1416]	# ang position of joint 3 in radians
  4  theta4	      u  [-3.1416,3.1416]	# ang position of joint 4 in radians
  5  theta5	      u  [-3.1416,3.1416]	# ang position of joint 5 in radians
  6  theta6	      u  [-3.1416,3.1416]	# ang position of joint 6 in radians
  7  theta7	      u  [-3.1416,3.1416]	# ang position of joint 7 in radians
  8  theta8	      u  [-3.1416,3.1416]	# ang position of joint 8 in radians
  9  y	      u  [0,Inf) # Cartesian distance of end point from position (0.1, 0.1, 0.1)

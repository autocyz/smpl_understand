'''
About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

'''

import sys
sys.path.append('/home/cyz/data/code/2d23d/SMPL/')
import smpl_webuser 
from smpl_webuser.serialization import load_model
import numpy as np

## Load SMPL model (here we load the female model)
## Make sure path is correct
m = load_model( '../../models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl' )

## Assign random pose and shape parameters


print(m.pose.size)

# m.pose[:] = np.random.rand(m.pose.size) * 3
# m.betas[:] = np.random.rand(m.betas.size) * 0.0
# m.pose[6] = 0.5

print(m.J[0:3])
print(type(m))
print("v_shaped: ", m.v_shaped[0:3, :])
print("v_posed: ", m.v_posed[0:3, :])
print("v_template: : ", m.v_template[0:3, :])
print("J_regressor_prior: ", m.J_regressor_prior[0:3, 0:3])

joints = np.dot(m.J_regressor.toarray(), m.r)
 # for i in range(m.pose.size):
#     m.pose[i] = 1
#     outmesh_path = './vertice/' + '%02d' % i + '.obj'
#     with open( outmesh_path, 'w') as fp:
#         for v in m.r:
#             fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
#
#         for f in m.f + 1:  # Faces are 1-based, not 0-based in obj files
#             fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
#     m.pose[i] = 0rint("J_regressor: ", m.J_regressor[0:3, 0:3])
## Write to an .obj file
outmesh_path = './vertices.obj'
with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    for f in m.f + 1:  # Faces are 1-based, not 0-based in obj files
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
print ('..Output mesh saved to: ', outmesh_path )


outjoint_path = './joints.obj'
with open(outjoint_path, 'w') as fp:
    for j in joints:
        fp.write('v %f %f %f\n' % (j[0], j[1], j[2]))
print ('..Output joints saved to: ', outjoint_path )

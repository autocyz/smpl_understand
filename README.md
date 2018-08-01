## SMPL
Offcial website: http://smpl.is.tue.mpg.de/     
Doc: http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf    
SMPL have 6890 vertices and 23+1 joints    

## SMPL model values: 
```
name: J_regressor_prior     type: <class 'scipy.sparse.csc.csc_matrix'>     size: (24, 6890)
name: pose                  type: <class 'chumpy.ch.Ch'>                    size: (72,)
name: f                     type: <type 'numpy.ndarray'>                    size: (13776, 3)
name: J_regressor           type: <class 'scipy.sparse.csc.csc_matrix'>     size: (24, 6890)
name: betas                 type: <class 'chumpy.ch.Ch'>                    size: (10,)
name: kintree_table         type: <type 'numpy.ndarray'>                    size: (2, 24)
name: J                     type: <class 'chumpy.reordering.transpose'>     size: (24, 3)
name: v_shaped              type: <class 'chumpy.ch_ops.add'>               size: (6890, 3)
name: weights_prior         type: <type 'numpy.ndarray'>                    size: (6890, 24)
name: trans                 type: <class 'chumpy.ch.Ch'>                    size: (3,)
name: v_posed               type: <class 'chumpy.ch_ops.add'>               size: (6890, 3)
name: weights               type: <class 'chumpy.ch.Ch'>                    size: (6890, 24)
name: vert_sym_idxs         type: <type 'numpy.ndarray'>                    size: (6890,)
name: posedirs              type: <class 'chumpy.ch.Ch'>                    size: (6890, 3, 207)
name: pose_training_info    type: <type 'dict'>                             size: 6
name: bs_style              type: <type 'str'>                              size: 3
name: v_template            type: <class 'chumpy.ch.Ch'>                    size: (6890, 3)
name: shapedirs             type: <class 'chumpy.ch.Ch'>                    size: (6890, 3, 10)
name: bs_type               type: <type 'str'>                              size: 7
name: r                     type: <type 'numpy.ndarray'>                    size: (6890, 3)
```

## Model inputs/outputs:
- inputs: pose, betas
- outputs: r, f          
change inputs, the model will be changed automaticly, authors use chumpy to complete this func. 

## Some equations about SMPL values
```
v_shaped = dot(shapedirs, betas)+v_template
J = dot(J_regressor, v_shaped) // different people have different shapes, different shapes have different joints

// joints about changed SMPL vertices
J_changed = dot(J_regressor, r)
```


import openpnm as op
import scipy as sp
from __transientAdvectionDiffusion__ import transientAdvectionDiffusion as tad

ws = op.core.Workspace()
ws.settings['local_data'] = True

########################## NETWORK ##########################
sp.random.seed(0)
pn = op.network.Cubic(shape=[40, 30, 1], spacing=1e-6, name='pn11')
#pn.add_boundary_pores()

########################## GEOMETRIES ##########################
geom = op.geometry.StickAndBall(network=pn, pores=pn.Ps, throats=pn.Ts)

########################## PHASES ##########################
water = op.phases.Water(network=pn)
water['throat.viscosity'] = water['pore.viscosity'][0]

########################## PHYSICS ##########################
phys_water = op.physics.GenericPhysics(network=pn, phase=water, geometry=geom)

# compute throat conductance
mod = op.physics.models.hydraulic_conductance.hagen_poiseuille
phys_water.add_model(propname='throat.conductance',
                     model=mod, viscosity='throat.viscosity')
phys_water.regenerate_models()

########################## ALGORITHMS ##########################

alg1=op.algorithms.GenericLinearTransport(network=pn,phase=water)
alg1.setup(conductance='throat.conductance',quantity='pore.pressure')

inlet = pn.pores('back')[0:7] # pore inlet
outlet = pn.pores('front')[0:7] # pore outlet
spacer = pn.pores('left')
for i in range (7):
    layer = pn.pores('left')+i+1 
    spacer = sp.append(spacer,layer)
spacer = sp.sort(spacer)

alg1.set_BC(pores=inlet, bctype='dirichlet', bcvalues=202650)
alg1.set_BC(pores=outlet, bctype='dirichlet', bcvalues=101325)
alg1['pore.pressure'] = 101325

# compute the pressure field
alg1.run()

########################## TRANS CONV DIFF ADS ##########################

alg2=tad(pn,geom)
alg2.set_diffusion(2.14e-5)
alg2.set_advection(alg1['pore.pressure'],phys_water['throat.conductance'])
alg2.set_BC(inlet,'dirichlet', 1)
alg2.set_BC(outlet,'dirichlet', 0)
alg2.set_timeSolver()
alg2.run()


#pn['pore.inlet']=sp.zeros((pn.Np),dtype='bool')
#pn['pore.inlet'][inlet]=1
#pn['pore.outlet']=sp.zeros((pn.Np),dtype='bool')
#pn['pore.outlet'][outlet]=1
#water['pore.concentration'] = alg2.b
#water['pore.pressure'] = alg1['pore.pressure']
#
#
#
#alg = op.algorithms.FickianDiffusion(network=pn, phase=water)
#alg.setup(conductance='throat.conductance', quantity='pore.mole_fraction')
#alg.set_BC(pores=inlet, bctype='dirichlet', bcvalues=0.5)
#alg.set_BC(pores=outlet, bctype='dirichlet', bcvalues=0.0)
#alg['pore.mole_fraction'] = 0
#alg.run()
#water['pore.mole_fraction']=alg['pore.mole_fraction']
#water['pore.diameter']=geom['pore.diameter']
#
#
#ws.export_data(simulation=ws['sim_001'],filename='vis_01',phases=water)


















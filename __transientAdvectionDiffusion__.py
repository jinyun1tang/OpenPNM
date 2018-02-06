import scipy as sp

class transientAdvectionDiffusion:

    def __init__(self, network, geometry):
        self.net = network
        self.geom = geometry
        self.diffusion = False
        self.advection = False
        self.bcs = []

    def set_diffusion(self, diffusionCoef):
        self.diffusion = True
        self.D = diffusionCoef

    def set_advection(self, pressure, hydraulicCond):
        self.advection = True
        self.p = pressure
        self.HC = hydraulicCond

    def set_timeSolver(self, simulationTime=10, timeStep=0.01):
        self.simTime = simulationTime
        self.deltaT = timeStep

    def build_A(self):
        diagA = sp.zeros((self.net.Np))
        A = sp.zeros((self.net.Np,self.net.Np))
        
        deltaT = self.deltaT # time step
        V = self.geom['pore.volume'] # pore volume
        tA = self.geom['throat.area'] # throat cross section area
        deltaL = self.geom['throat.length'] # throat length
        nt = self.net.find_neighbor_throats(pores=self.net['pore.all'], \
             flatten=False, mode='not_intersection') # pore neighbor throats

        if ((self.diffusion==True) and (self.advection==True)):
            HC = self.HC # throat hydraulic conductance
            p = self.p # pore pressure
            D = self.D # diffusion coefficient
            for i in range (self.net.Np):
                diagA[i] = 1+(deltaT/V[i])* sp.sum( (1./deltaL[nt[i]]) * (\
                (((V[i]*HC[nt[i]])/tA[nt[i]])*(p[i]-p[self.net.find_neighbor_pores(i)]))- \
                (D*tA[nt[i]])) )
            
                j1 = self.net['throat.conns'][nt[i]]
                j2 = sp.reshape(j1,sp.size(j1))
                j = j2[j2!=i]
    
                A[i,j] = -(deltaT/V[i])* ((1./deltaL[nt[i]]) * (\
                (((V[i]*HC[nt[i]])/tA[nt[i]])*(p[i]-p[self.net.find_neighbor_pores(i)]))- \
                (D*tA[nt[i]])) )

        elif ((self.diffusion==True) and (self.advection==False)):
            D = self.D # diffusion coefficient
            for i in range (self.net.Np):
                diagA[i] = 1+(deltaT/V[i])* sp.sum( (1./deltaL[nt[i]]) * ( \
                - (D*tA[nt[i]])) )
            
                j1 = self.net['throat.conns'][nt[i]]
                j2 = sp.reshape(j1,sp.size(j1))
                j = j2[j2!=i]
    
                A[i,j] = -(deltaT/V[i])* ((1./deltaL[nt[i]]) * ( \
                - (D*tA[nt[i]])) )

        elif ((self.diffusion==False) and (self.advection==True)):
            HC = self.HC # throat hydraulic conductance
            p = self.p # pore pressure
            for i in range (self.net.Np):
                diagA[i] = 1+(deltaT/V[i])* sp.sum( (1./deltaL[nt[i]]) * \
                (((V[i]*HC[nt[i]])/tA[nt[i]])*(p[i]-p[self.net.find_neighbor_pores(i)])) )
            
                j1 = self.net['throat.conns'][nt[i]]
                j2 = sp.reshape(j1,sp.size(j1))
                j = j2[j2!=i]
    
                A[i,j] = -(deltaT/V[i])* ((1./deltaL[nt[i]]) * \
                (((V[i]*HC[nt[i]])/tA[nt[i]])*(p[i]-p[self.net.find_neighbor_pores(i)])) )

        if (self.bcs!=[]):
            for i in range (len(self.bcs)):
                if (self.bcs[i]['BCtype']=='dirichlet'):
                    A[self.bcs[i]['pores'],:] = 0
                    diagA[self.bcs[i]['pores']] = 1
        sp.fill_diagonal(A,diagA)
        self.A = sp.sparse.csr_matrix(A)
        return A

    def build_b(self):
        b = sp.zeros((self.net.Np)) # pore initial concentration
        if (self.bcs!=[]):
            for i in range (len(self.bcs)):
                b[self.bcs[i]['pores']] = self.bcs[i]['value']
        self.b = b            
        return b

    def set_BC(self, pores, BCtype, value):
        self.bcs.append({'pores':pores, 'BCtype':BCtype, 'value':value})
        
    def run(self):
        self.build_A()
        self.build_b()
        self.solve()

    def solve(self):
        A = self.A
        b_0 = self.b
        time = sp.arange(self.deltaT,self.simTime+self.deltaT,self.deltaT)
        print('Transient solver started, t = 0 s, ...')
        for t in (time):
            b_0 = sp.sparse.linalg.spsolve(A,b_0)
        self.b = b_0
        print('Transient solver finished, t = ',time[-1],'s.')
        return b_0


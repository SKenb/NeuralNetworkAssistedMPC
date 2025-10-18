import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

class FlowReactor:
    def __init__(self, length, diameter, reactionNetworkCallback=None, dispersionCoefficient=0, setSpaceSamples=500):
        self.diameter = diameter
        self.setLength(length)
        self.setDispersionCoefficient(dispersionCoefficient)
        self.setSpaceSamples(setSpaceSamples)
        self.reactionNetworkCallback = reactionNetworkCallback

        self.lastSimResult = None
        
    def getReactorCrossSection(self):
        return np.pi * ((self.diameter / 2) ** 2)

    def setLength(self, length):
        self.length = length
        self.volume = self.getReactorCrossSection() * self.length

    def setDiameter(self, diameter):
        self.diameter = diameter
        self.setLength(self.length)

    def getVolumeIn_mL(self):
        return self.volume * 1e6

    def setDispersionCoefficient(self, dispersionCoefficient):
        self.dispersionCoefficient = dispersionCoefficient

    def setSpaceSamples(self, setSpaceSamples):
        self.xSamples = setSpaceSamples
        
    def getNumberOfSpaceSamples(self):
        return self.xSamples
    
    def getReactorVolumeInLiters(self):
        return self.volume * 1000   # m^3 --- L
    
    def volumetricFlowRateToFlowRate(self, volumetricFlowRate):
        x = volumetricFlowRate / 60 * 1e-3          # mL / min --- L / s
        x = x * 1e-3                                # L / s ------ m^3 / s
        return x / self.getReactorCrossSection()    # m^3 / s ---- m / s

    def getReactorSpaceSamples(self):
        dx = self.length / self.xSamples
        return np.arange(0, self.length + dx, dx)
    
    def getProposedDeltaT(self, maxFlowRate):
        maxFlowRate = np.fmax(maxFlowRate, 1e-3)
        dx = self.length / self.xSamples
        return dx / maxFlowRate / 2
    
    def getTimeVector(self, Tend, maxFlowRate):
        return np.arange(0, Tend, self.getProposedDeltaT(maxFlowRate))

    def __str__(self):
        return f"FlowReactor (volume={1000*self.getReactorVolumeInLiters()} mL, length={self.length} m, diameter={1000*self.diameter} mm, D={self.dispersionCoefficient})"

    def setReactionNetworkCallback(self, reactionNetworkCallback):
        self.reactionNetworkCallback = reactionNetworkCallback

    def plot(self, Cin=None, figure=None, showPlot=False, measuredCout=None, measuredCoutTimeVec=None, CinTimeVec=None, additionalTracesToPlot=None):
        if figure is None: figure = plt.figure()
        self.plotSpaceTime(figure=figure, rows=2, showPlot=False)
        self.plotOutputConcentration(Cin=Cin, measuredCout=measuredCout, measuredCoutTimeVec=measuredCoutTimeVec, CinTimeVec=CinTimeVec, additionalTracesToPlot=additionalTracesToPlot, figure=figure, rows=2, rowIdx=2, showPlot=showPlot)

    def plotSpaceTime(self, title="Reactor Time Evolution (3D)", figure=None, rows=1, rowIdx=1, showPlot=False):
        if self.lastSimResult is None: return
        time, Cspatial = self.lastSimResult
        [nSpecies, _, _] = Cspatial.shape

        if figure is None: figure = plt.figure()

        X, Y = np.meshgrid(time, self.getReactorSpaceSamples())

        for i in range(nSpecies):
            ax = figure.add_subplot(rows, nSpecies, ((nSpecies * (rowIdx-1)) + i+1), projection='3d')

            surf = ax.plot_surface(X, Y, Cspatial[i, :, :], cmap='viridis')

            ax.set_title(title)
            ax.set_xlabel("Time t in s")
            ax.set_ylabel("Length x in m")
            ax.set_zlabel("Concentration")
            ax.view_init(elev=90, azim=0)
            ax.set_zlim(0, 1.1)


        figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        if showPlot: plt.show()

    def plotOutputConcentration(self, Cin=None, labelCout="Cout", measuredCout=None, measuredCoutTimeVec=None, CinTimeVec=None, additionalTracesToPlot=None, title="Reactor Time Evolution (2D)", figure=None, rows=1, rowIdx=1, showPlot=True):
        if self.lastSimResult is None: return
        time, Cspatial = self.lastSimResult
        [nSpecies, _, _] = Cspatial.shape

        if figure is None: figure = plt.figure()

        for i in range(nSpecies):
            ax = figure.add_subplot(rows, nSpecies, ((nSpecies * (rowIdx-1)) + i+1))

            if Cin is not None: 
                Cin_i = Cin[i, :]

                if CinTimeVec is not None:
                    CinInterp = interp1d(CinTimeVec, Cin[i, :], kind='linear', fill_value="extrapolate")
                    Cin_i = CinInterp(time)

                ax.plot(time, Cin_i, label="Cin", linestyle='--')
            
            if measuredCout is not None: 
                measuredCout_i = measuredCout[i, :]

                if measuredCoutTimeVec is not None:
                    measuredCoutInterp = interp1d(measuredCoutTimeVec, measuredCout[i, :], kind='linear', fill_value="extrapolate")
                    measuredCout_i = measuredCoutInterp(time)
                
                if not np.all(np.isnan(measuredCout_i)):
                    ax.plot(time, measuredCout_i, label="measured Cout", linestyle='--')


            for trace in additionalTracesToPlot or []:
                x, y = trace.get("x"), trace.get("y")
                label, linestyle = trace.get("label", ""), trace.get("linestyle", "-")
                plotType = trace.get("plotType", "line")

                y_ = y
                if y.ndim > 1: y_ = y[i, :] 

                YInterp = interp1d(x, y_, kind='linear', fill_value="extrapolate")
                YInterp_i = YInterp(time)
                
                if not np.all(np.isnan(YInterp_i)):
                    if plotType == "stddev":
                        z = trace.get("z")
                        if z is not None:   
                            YStdDevInterp = interp1d(x, z[i, :], kind='linear', fill_value="extrapolate")
                            YStdDevInterp_i = YStdDevInterp(time)

                            ax.fill_between(time, YInterp_i - YStdDevInterp_i, YInterp_i + YStdDevInterp_i, alpha=0.2, label=label)
                    else:
                        ax.plot(time, YInterp_i, label=label, linestyle=linestyle)

            #if CoutPredicted is not None and CoutPredictedTime is not None:
            #    CoutPredictedInterp = interp1d(CoutPredictedTime, CoutPredicted[i, :], kind='linear', fill_value="extrapolate")
            #    CoutPredicted_i = CoutPredictedInterp(time)
            #    
            #    if not np.all(np.isnan(CoutPredicted_i)):
            #        if CoutPredictedStdDev is not None:
            #            CoutPredictedStdDevInterp = interp1d(CoutPredictedTime, CoutPredictedStdDev[i, :], kind='linear', fill_value="extrapolate")
            #            CoutPredictedStdDev_i = CoutPredictedStdDevInterp(time)
            #            
            #            ax.fill_between(time, CoutPredicted_i - CoutPredictedStdDev_i, CoutPredicted_i + CoutPredictedStdDev_i, alpha=0.2, label="Cout predicted std dev")
            #            ax.plot(time, CoutPredicted_i, label="Cout predicted", linestyle='--')
            #        else:
            #            ax.plot(time, CoutPredicted_i, label="Cout predicted", linestyle='--')
            #
            
            ax.plot(time, Cspatial[i, -1, :], label=labelCout)
            ax.set_title(title)
            ax.set_xlabel("Time t in s")
            ax.set_ylabel("Concentration in mol/L")
            ax.set_ylim(0, 1.1)

            ax.legend()

        if showPlot: plt.show()

    def resetLastSimResult(self):
        self.lastSimResult = None

    def getLastSimResult(self):
        return self.lastSimResult
    
    def setLastSimResult(self, lastSimResult):
        self.lastSimResult = lastSimResult
        
    def simulateStep(self, startTime, endTime, time_vec, Cin, flowRate, temperature=20, isVolumetricFlowRate=True, method="RK45", rtol=1e-6, atol=1e-6, Cspatial0=None):
        if isVolumetricFlowRate: flowRate = self.volumetricFlowRateToFlowRate(flowRate)        
        
        if Cspatial0 is None:
            if self.lastSimResult is not None:
                _, prevCspatial = self.lastSimResult
                Cspatial0 = prevCspatial[:, 1:, -1]
        
        input_fcn = interp1d(time_vec, Cin, kind='linear', fill_value="extrapolate")
        temperature_fcn = (lambda _: temperature) if np.isscalar(temperature) else interp1d(time_vec, temperature, kind='linear', fill_value="extrapolate")
        flowRate_fcn = (lambda _: flowRate) if np.isscalar(flowRate) else interp1d(time_vec, flowRate, kind='linear', fill_value="extrapolate")

        proposedDeltaT = self.getProposedDeltaT(flowRate if np.isscalar(flowRate) else flowRate.max())
        minDeltaT = (endTime - startTime) / 100
        time_vec_step = np.arange(startTime, endTime, np.fmin(proposedDeltaT, minDeltaT))

        Cin = input_fcn(time_vec_step)
        temperature = temperature if np.isscalar(temperature) else temperature_fcn(time_vec_step)
        flowRate = flowRate if np.isscalar(flowRate) else flowRate_fcn(time_vec_step)

        return self.simulate(time_vec_step, Cin, flowRate, isVolumetricFlowRate=False, temperature=temperature, Cspatial0=Cspatial0, method=method, rtol=rtol, atol=atol)

    def simulate(self, time_vec, Cin, flowRate, temperature=20, isVolumetricFlowRate=True, Cspatial0=None, method="RK45", rtol=1e-6, atol=1e-6):
        if isVolumetricFlowRate: flowRate = self.volumetricFlowRateToFlowRate(flowRate)  
        
        def get_dCdt(t, Cvec, nSpecies, dx, time_vec, Cin, flowRate, temperature):
            # Csvec = [A1, A2, A3, ... AN, B1, B2, B3, ... BN, ...] 
            # A = C1, B = C2, ...
            # A1 at pos 1

            dCdx = np.zeros_like(Cvec)
            dC2d2x = np.zeros_like(Cvec)
            rCi = np.zeros_like(Cvec)

            # It might be, that the ivp_solve also uses time samples in between the time samples of Cin
            input_fcn = interp1d(time_vec, Cin, kind='linear', fill_value="extrapolate")
            temperature_fcn = (lambda _: temperature) if np.isscalar(temperature) else interp1d(time_vec, temperature, kind='linear', fill_value="extrapolate")
            flowRate_fcn = (lambda _: flowRate) if np.isscalar(flowRate) else interp1d(time_vec, flowRate, kind='linear', fill_value="extrapolate")

            for i in range(nSpecies): 
                sIdx = i * self.xSamples
                eIdx = (i + 1) * self.xSamples

                # Convection
                Ci_spatial = Cvec[sIdx:eIdx]
                dCdx[sIdx+1:eIdx] = (Ci_spatial[1:] - Ci_spatial[:-1]) / dx
                dCdx[sIdx] = (Ci_spatial[0] - input_fcn(t)[i]) / dx

                # Dispersion
                dC2d2x[sIdx] = (input_fcn(t)[i] - 2*Ci_spatial[0] + Ci_spatial[1]) / dx**2

                d2Cidx2_center = (Ci_spatial[:-2] - 2*Ci_spatial[1:-1] + Ci_spatial[2:]) / dx**2
                d2Cidx2_end = (Ci_spatial[-2] - Ci_spatial[-1]) / dx**2
                dC2d2x[sIdx+1:eIdx] = np.append(d2Cidx2_center, d2Cidx2_end)

                # Reaction
                if self.reactionNetworkCallback is not None:
                    rCi[sIdx:eIdx] = self.reactionNetworkCallback(Cvec.reshape(nSpecies, self.xSamples), i, temperature_fcn(t))

            return self.dispersionCoefficient * dC2d2x -1 * flowRate_fcn(t) * dCdx + rCi

        # Multiple concentration inputs
        # Cin [Species Idx x time]
        # reactionNetworkCallback: Cspatial_t-1 
        #   [nSpecies x nReactorSamples], speciesIdx, temperature [in °C] -> reactionRates i [nReactorSamples] 
        nSpecies = Cin.shape[0]
        dx = self.length / self.xSamples

        Cvec0 = np.zeros((nSpecies*self.xSamples))
        if Cspatial0 is not None:
            assert Cspatial0.shape == (nSpecies, self.xSamples)
            Cvec0 = Cspatial0.reshape(nSpecies*self.xSamples)

        # t_eval specifies the time points where you want the solver to output results.
        # The solver still internally takes as many steps as needed (adaptive stepping) to maintain accuracy, even between the specified t_eval points.
        # You can choose t_eval to include times of known discontinuities or events of interest while letting the solver handle the intermediate computations adaptively.
        derivative_f = lambda t_, x_: get_dCdt(t_, x_, nSpecies, dx, time_vec, Cin, flowRate, temperature)
        solution = solve_ivp(derivative_f, (time_vec[0], time_vec[-1]), Cvec0, t_eval=time_vec, method=method, rtol=rtol, atol=atol)
        
        time = solution.t
        Cspatial = solution.y.reshape(nSpecies, self.xSamples, -1)
        
        CinInterpolated = interp1d(time_vec, Cin, kind='linear', fill_value="extrapolate")
        Cspatial = np.concatenate((CinInterpolated(time)[:, np.newaxis,:], Cspatial), axis=1)

        # Cspatial = [nSpecies x nReactorSamples x nTimeSamples]
        Cout = Cspatial[:, -1, :]

        self.lastSimResult = (time, Cspatial)
        return time, Cout, Cspatial
    
    def simulateExplicitEuler(self, time_vec, Cin, flowRate, temperature=20, isVolumetricFlowRate=True, Cspatial0=None):
        # Multiple concentration inputs
        # Cin [Species Idx x time]
        # reactionNetworkCallback: Cspatial_t-1 
        #   [nSpecies x nReactorSamples], speciesIdx, temperature [in °C] -> reactionRates i [nReactorSamples] 
        if isVolumetricFlowRate: flowRate = self.volumetricFlowRateToFlowRate(flowRate)  
        [nSpecies, nTimeSamples] = Cin.shape

        dt = time_vec[1] - time_vec[0]
        dx = self.length / self.xSamples

        # speciesIdx x time x space - where space idx 0 = inlet thus +1
        Cspatial = np.zeros((nSpecies, nTimeSamples, self.xSamples+1))
        Cspatial[:, :, 0] = Cin[:, :]
        if Cspatial0 is not None:
            assert Cspatial0.shape == (nSpecies, self.xSamples)
            Cspatial[:, 0, 1:] = Cspatial0

        
        # Time loop
        for t in range(nTimeSamples):
            if t <= 0: continue # skip first time step x0 = 0

            temperature_t = temperature if np.isscalar(temperature) else temperature[t]
            flowRate_t = flowRate if np.isscalar(flowRate) else flowRate[t]
    
            for i in range(nSpecies):

                # Dispersion
                d2Cidx2_center = (Cspatial[i, t-1, :-2] - 2*Cspatial[i, t-1, 1:-1] + Cspatial[i, t-1, 2:]) / dx**2
                d2Cidx2_end = (Cspatial[i, t-1, -2] - Cspatial[i, t-1, -1]) / dx**2
                d2Cidx2 = np.append(d2Cidx2_center, d2Cidx2_end)

                # Convection
                dCidx = (Cspatial[i, t-1, 1:] - Cspatial[i, t-1, :-1]) / dx

                # Reaction
                rCi = np.zeros(self.xSamples)

                if self.reactionNetworkCallback is not None:
                    rCi = self.reactionNetworkCallback(Cspatial[:, t-1, 1:], i, temperature_t)

                # Update concentration
                Cspatial[i, t, 1:] = Cspatial[i, t-1, 1:] + dt * (self.dispersionCoefficient * d2Cidx2 - flowRate_t * dCidx + rCi)


        # Cspatial = [nSpecies x nReactorSamples x nTimeSamples]
        Cspatial = Cspatial.transpose(0, 2, 1)
        Cout = Cspatial[:, -1, :]


        self.lastSimResult = (time_vec, Cspatial)
        return time_vec, Cout, Cspatial
    
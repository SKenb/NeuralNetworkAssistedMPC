import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import pickle
import imageio
import os

class SimulationInstances:
    class Instance:
        def __init__(self, ID, time_vec, Cin, flowRate, temperature, time_out, Cout, Cspatial, reactorSpaceSamples, simTime):
            self.ID = ID
            
            self.time_vec = time_vec
            self.Cin = Cin
            self.flowRate = flowRate
            self.temperature = temperature
            
            self.time_out = time_out
            self.Cout = Cout
            self.Cspatial = Cspatial
            
            self.reactorSpaceSamples = reactorSpaceSamples
            self.simTime = simTime
            
        def plot(self, showPlot=False):
            figure1 = plt.figure()
            nSpecies = self.Cout.shape[0]
            
            X, Y = np.meshgrid(self.time_out/60, self.reactorSpaceSamples)
            for i in range(nSpecies):
                ax = figure1.add_subplot(2, nSpecies, i+1, projection='3d')
                ax.plot_surface(X, Y, self.Cspatial[i, :, :], cmap='viridis')
                ax.set_xlabel("Time t in min")
                ax.set_ylabel("Length x in m")
                ax.set_zlabel("Concentration")
                ax.view_init(elev=90, azim=0)
                ax.set_zlim(0, 1.1)
                
                ax = figure1.add_subplot(2, nSpecies, nSpecies + i+1)
                if self.Cin is not None: ax.plot(self.time_vec/60, self.Cin[i, :], label="Cin", linestyle='--')
                ax.plot(self.time_out/60, self.Cspatial[i, -1, :], label="Cout")
                ax.set_xlabel("Time t in min")
                ax.set_ylabel("Concentration in mol/L")
                ax.set_ylim(0, 1.1)
                ax.legend()
                
            
            figure2 = plt.figure()
            ax = figure2.add_subplot(3, 1, 1)
            for i in range(4):
                ax.plot(self.time_vec/60, self.Cin[i, :], label=f"C{i}")
            ax.set_xlabel("Time t in min")
            ax.set_ylabel("Concentration in mol/L")
            
            ax = figure2.add_subplot(3, 1, 2)
            ax.plot(self.time_vec/60, self.flowRate)
            ax.set_xlabel("Time t in min")
            ax.set_ylabel("Flow rate in mL/min")
            
            ax = figure2.add_subplot(3, 1, 3)
            ax.plot(self.time_vec/60, self.temperature)
            ax.set_xlabel("Time t in min")
            ax.set_ylabel("Temperature in °C")
            
            if showPlot: plt.show()
            
        def toDataArray(self):
            return {
                "ID": self.ID,
                "time_vec": self.time_vec,
                "Cin": self.Cin,
                "flowRate": self.flowRate,
                "temperature": self.temperature,  
                "time_out": self.time_out,
                "Cout": self.Cout,
                "Cspatial": self.Cspatial,
                "reactorSpaceSamples": self.reactorSpaceSamples,
                "simTime": self.simTime
            }
        
        def getTrainingData(self, timeBack, timePrediction, sampleTimeNNInput, trainingSamplesPerSimulation=100, includeCoutSpecies=None, predictSpecies=None):
            # timeBack ... time interval back in time which used for prediction
            # timePrediction ... time interval which is predicted
            tBlock = (timeBack + timePrediction)

            tIndices = np.arange(self.time_vec[0], self.time_vec[-1] - tBlock, trainingSamplesPerSimulation)

            X, Y = None, None
            for tStart in tIndices:
                tBlckStart, t, tBlckEnd = tStart, (tStart + timeBack), (tStart + timeBack + timePrediction)
    
                tNN_in = np.arange(tBlckStart, t, sampleTimeNNInput)
                tNN_out = np.arange(t, tBlckEnd, sampleTimeNNInput)

                tIn_idx = np.where((self.time_vec >= tBlckStart) & (self.time_vec <= t))
                tOut_idx = np.where((self.time_vec > t) & (self.time_vec <= tBlckEnd))
        
                cin = self.Cin[0:2, tIn_idx][:, 0, :]
                c_combi = np.array(cin[0, :]) * np.array(cin[1, :])
                flr = self.flowRate[tIn_idx]
                temp = self.temperature[tIn_idx]
                cout = self.Cout[:, tIn_idx][:, 0, :]

                inp_int_fcn = interp1d(self.time_vec[tIn_idx], np.vstack((cin, c_combi, flr, temp)), fill_value="extrapolate")
                cout_int_fcn = interp1d(self.time_out[tIn_idx], cout, fill_value="extrapolate")

                x = inp_int_fcn(tNN_in)
                if includeCoutSpecies is not None:
                    x = np.vstack((x, cout_int_fcn(tNN_in)[includeCoutSpecies]))

                y_int_fcn = interp1d(self.time_out[tOut_idx], self.Cout[:, tOut_idx][:, 0, :], fill_value="extrapolate")
                y = y_int_fcn(tNN_out)

                if predictSpecies is not None:
                    y = y[predictSpecies]

                X = x.flatten() if X is None else np.vstack((X, x.flatten()))
                Y = y.flatten() if Y is None else np.vstack((Y, y.flatten()))

            return X, Y

            
        # end def
    
    def __init__(self, name, reactorSpaceSamples):
        self.name = name
        self.reactorSpaceSamples = reactorSpaceSamples
        self.instances = []
        
    def append(self, ID, time_vec, Cin, flowRate, temperature, time_out, Cout, Cspatial, simTime=None):
        if ID is None: ID = np.random.randint(10000, 99999)
        inst = SimulationInstances.Instance(ID, time_vec, Cin, flowRate, temperature, time_out, Cout, Cspatial, self.reactorSpaceSamples, simTime)
        self.instances.append(inst)
        
    def save(self, fileName):  
        data = []
        for inst in self.instances:
            data.append(inst.toDataArray())
            
        with open(fileName, "wb") as file:
            pickle.dump({
                "name": self.name, 
                "reactorSpaceSamples": self.reactorSpaceSamples,
                "instances": data
            }, file)
            
    def load(self, fileName, loadCspatial=False):
        with open(fileName, "rb") as file: 
            data = pickle.load(file)
            
            self.name = data["name"]
            self.reactorSpaceSamples = data["reactorSpaceSamples"]
            
            for d_ in data["instances"]: 
                self.append(d_["ID"], d_["time_vec"], d_["Cin"], d_["flowRate"], d_["temperature"], d_["time_out"], d_["Cout"], (d_["Cspatial"] if loadCspatial else None))

    def getTrainingData(self, timeBack, timePrediction, sampleTimeNNInput, trainingSamplesPerSimulation, includeCoutSpecies=None, predictSpecies=None, normalizingMethod='minmax'):
        X, Y = None, None

        for inst_ in self.instances:
            X_, Y_ = inst_.getTrainingData(
                timeBack, 
                timePrediction, 
                sampleTimeNNInput, 
                trainingSamplesPerSimulation,
                includeCoutSpecies,
                predictSpecies
            )

            X = X_ if X is None else np.vstack((X, X_))
            Y = Y_ if Y is None else np.vstack((Y, Y_))

        return X, Y
    
    
class ReferenceSignals:
    def __init__(self, nOutputSpecies, tEnd, sampleTime):
        self.sampleTime = sampleTime
        self.time = np.arange(0, tEnd, sampleTime)
        self.Cref = np.nan * np.ones((nOutputSpecies, self.time.shape[0]))

    def setReference(self, idx, Ci_ref):
        self.Cref[idx, :] = Ci_ref

    def setReferenceWhereTimeGreater(self, idx, timePoint, Ci_ref):
        self.Cref[idx, np.where(self.time >= timePoint)] = Ci_ref

    def plot(self, figure=None, showPlot=False):
        if figure is None: figure = plt.figure()
        
        nSpecies = self.Cref.shape[0]
        
        for idx in range(nSpecies):
            ax = figure.add_subplot(1, nSpecies, idx+1)
            ax.plot(self.time, self.Cref[idx, :])
            ax.set_title(f"Reference C{idx+1}")
            ax.set_xlabel("time t in s")
            ax.set_ylabel("Concentration")
            
        if showPlot:
            plt.show()

    def getReferences(self, timePoints):
        int_fcn = interp1d(self.time, self.Cref, kind='linear', fill_value="extrapolate")
        return int_fcn(timePoints)

class EstimationHist:
    def __init__(self):
        self.time = np.array([])
        self.estCspatial = np.array([])
        self.Cspatial = np.array([])
        
        self.mse = np.array([])
        
    def append(self, time_, estCspatial_, Cspatial_):   
        self.time = np.append(self.time, time_)
        self.estCspatial = np.array([estCspatial_]) if self.estCspatial.size == 0 else np.concatenate((self.estCspatial, np.array([estCspatial_])), axis=0)
        self.Cspatial = np.array([Cspatial_]) if self.Cspatial.size == 0 else np.concatenate((self.Cspatial, np.array([Cspatial_])), axis=0)
        
        self._calcAndAppendMSE(estCspatial_, Cspatial_)
        
    def _calcAndAppendMSE(self, estCspatial_, Cspatial_):
        diff = estCspatial_ - Cspatial_
        mse_ = np.mean(diff**2, axis=1)
        
        self.mse = np.array([mse_]) if self.mse.size == 0 else np.concatenate((self.mse, np.array([mse_])), axis=0)
        
    def plot(self, figure=None, showPlot=False):
        if figure is None: figure = plt.figure()    
        nSpecies = self.mse.shape[1]
        
        for idx in range(nSpecies):
            ax = figure.add_subplot(1, nSpecies, idx+1)
            ax.plot(self.time, self.mse[:, idx])
            ax.set_title(f"MSE C{idx+1}")
            ax.set_xlabel("time t in s")
            ax.set_ylabel("MSE")
            
        if showPlot:
            plt.show()

class OutputHist:
    def __init__(self):
        self.time = np.array([])
        self.Cout = np.array([])
        self.Cspatial = np.array([])
        
    def append(self, time_, Cout_, Cspatial_):   
        self.time = np.append(self.time, time_)
        self.Cout = Cout_ if self.Cout.size == 0 else np.concatenate((self.Cout, Cout_), axis=1)
        self.Cspatial = Cspatial_ if self.Cspatial.size == 0 else np.concatenate((self.Cspatial, Cspatial_), axis=2)

    def getCoutForTimeVec(self, timeVec, forcedSizeofZeros=4):
        if self.Cout.size == 0: return np.zeros((forcedSizeofZeros, timeVec.shape[0]))
        return interp1d(self.time, self.Cout, fill_value="extrapolate", axis=1)(timeVec)

    def plot(self, figure=None, showPlot=False, refSignals=None, onlyPositiveTime=True, addLimits=True):
        if figure is None: figure = plt.figure()
    
        nSpecies = self.Cout.shape[0]
        X, Y = np.meshgrid(self.time, np.linspace(0, 1, self.Cspatial.shape[1]))
        
        for i in range(nSpecies):
            ax = figure.add_subplot(2, nSpecies, i+1, projection='3d')
            ax.plot_surface(X, Y, self.Cspatial[i, :, :], label=f"C{i+1}")
            ax.set_title(f"Cspatial C{i+1}")
            ax.set_xlabel("time t in s")
            ax.set_ylabel("Norm. position x in m")
            ax.set_zlabel("Concentration")

            time = self.time
            Cout = self.Cout[i, :]

            if onlyPositiveTime:
                time_mask = self.time >= 0
                time = self.time[time_mask]
                Cout = self.Cout[i, time_mask]
            
            ax = figure.add_subplot(2, nSpecies, nSpecies+i+1)
            ax.plot(time/60, Cout, label=f"C{i+1}")
            if refSignals is not None:
                ax.plot(refSignals.time/60, refSignals.Cref[i, :], label=f"Ref C{i+1}", linestyle='--')
            ax.set_title(f"Output C{i+1}")
            ax.set_xlabel("time t in min")
            ax.set_ylabel("Concentration")
            ax.legend()
            if addLimits: ax.set_ylim((0, 1))
      
        if showPlot: plt.show()  

    def createGif(self, timeVecCref, Cref):
        x = np.linspace(0, 1, self.Cspatial.shape[1])

        filenames = []
        skipIdx = 30
        for idx, t in enumerate(self.time[::skipIdx]):

            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(3, 3, figure=fig)  # 3 Zeilen, 3 Spalten

            for row in range(3):

                data = self.Cspatial[row, :, skipIdx * idx]

                # Heatmap über Spalte 0 und 1
                ax1 = fig.add_subplot(gs[row, :2])
                heat = ax1.imshow(
                    np.array([data, data]), aspect='auto', cmap='viridis',
                    extent=[x.min(), x.max(), 0, .8],
                    vmin=0, vmax=1
                )
                ax1.set_title(f"Reactor C{row+1} (Time: {np.round(t/60)} min)")
                ax1.set_xlabel("Position")
                ax1.set_yticks([])
                plt.colorbar(heat, ax=ax1)

                # Zeitverlauf in Spalte 2
                timeVec = self.time[self.time <= t]
                ax2 = fig.add_subplot(gs[row, 2])
                ax2.plot(timeVec / 60, self.Cout[row, self.time <= t], color='blue')

                Cref_i = Cref[row, :]
                if not np.any(np.isnan(Cref_i)):
                    ax2.plot(timeVecCref[timeVecCref <= t]/60, Cref_i[timeVecCref <= t], '--', color='orange', label='Cref')
                    ax2.legend()

                ax2.set_title(f"C{row+1}out")
                ax2.set_xlabel("Time (min)")
                ax2.set_ylim([0, 1])

            plt.tight_layout()
            
            fname = f"./Tmp/frame_{t}.png"
            plt.savefig(fname)
            filenames.append(fname)
            plt.close()

            print(f"\t[GIF] Generated and saved image {idx} / {len(self.time[::skipIdx])}")

        # GIF erzeugen
        with imageio.get_writer('1d_heatmap.gif', mode='I', duration=0.1, loop=0) as writer:
            for fname in filenames:
                writer.append_data(imageio.imread(fname))

        # Aufräumen
        for fname in filenames:
            os.remove(fname)

    
class InputHist:
    def __init__(self):
        self.time = np.array([])
        self.Cin = np.array([])
        self.flowRate = np.array([])
        self.temperature = np.array([])

    def getCinForTimeVec(self, timeVec, forcedSizeofZeros=4):
        if self.Cin.size == 0: return np.zeros((forcedSizeofZeros, timeVec.shape[0]))
        interpolatedValues = interp1d(self.time, self.Cin, fill_value="extrapolate", axis=1)(timeVec)
        interpolatedValues[:, timeVec < 0] = 0
        return interpolatedValues

    def getFlowRateForTimeVec(self, timeVec):
        if self.flowRate.size == 0: return 0*timeVec
        interpolatedValues = interp1d(self.time, self.flowRate, fill_value="extrapolate", axis=0)(timeVec)
        interpolatedValues[timeVec < 0] = 0
        return interpolatedValues
    
    def getTemperatureForTimeVec(self, timeVec):
        if self.temperature.size == 0: return 0*timeVec
        interpolatedValues = interp1d(self.time, self.temperature, fill_value="extrapolate", axis=0)(timeVec)
        interpolatedValues[timeVec < 0] = 0
        return interpolatedValues
    
    def append(self, time_, Cin_, flowRate_, temperature_):
        # Keep only entries where existing time is >= incomingMinTime
        if self.time.shape[0] > 0:
            keep_indices = self.time < time_[0]
            self.time = self.time[keep_indices]
            self.Cin = self.Cin[:, keep_indices] if self.Cin.size != 0 else self.Cin
            self.flowRate = self.flowRate[keep_indices]
            self.temperature = self.temperature[keep_indices]
        
        self.time = np.append(self.time, time_)
        self.Cin = Cin_ if self.Cin.size == 0 else np.concatenate((self.Cin, Cin_), axis=1)
        self.flowRate = np.append(self.flowRate, flowRate_)
        self.temperature = np.append(self.temperature, temperature_)
        
    def plot(self, figure=None, showPlot=False, addLimits=True):
        if figure is None: figure = plt.figure()
        
        ax = figure.add_subplot(1, 3, 1)
        for i in range(self.Cin.shape[0]):
            ax.plot(self.time, self.Cin[i, :], label=f"C{i+1}")
        ax.set_title("Cin")
        ax.set_xlabel("time")
        ax.legend()
        if addLimits: ax.set_ylim((0, 1))
        
        ax = figure.add_subplot(1, 3, 2)
        ax.plot(self.time, self.flowRate)
        ax.set_title("flowRate")
        ax.set_xlabel("time")
        if addLimits: ax.set_ylim((0, 2))
        
        ax = figure.add_subplot(1, 3, 3)
        ax.plot(self.time, self.temperature)
        ax.set_title("temperature")
        ax.set_xlabel("time")
        
        if showPlot:
            plt.show()

class NNPredictionHist:
    def __init__(self):
        self.time = np.array([])
        self.allCout = np.array([])
        
    def append(self, time_, predictedCout):
        self.time = np.vstack((self.time, time_)) if self.time.size > 0 else time_
        self.allCout = np.vstack((self.allCout, predictedCout)) if self.allCout.size > 0 else predictedCout

    def getLastPredictionValues(self):
        if self.allCout.size == 0: return None, None    
        return self.time[:, -1], self.allCout[:, -1]

    def getMeanForTime(self, time):
        unique_times, mean_values, _ = self.getMeanAndStd()
        if unique_times is None: return None, None

        return interp1d(unique_times, mean_values)(time)

    def getMeanAndStd(self):
        if self.allCout.size == 0: return None, None, None
        
        # Flatten the time and predictions for aggregation
        time = self.time.flatten()
        flat_Cout = self.allCout.flatten()
        
        # Aggregate data by unique time points
        unique_times = np.unique(time)
        mean_values = []
        std_devs = []
        
        for t in unique_times:
            # Find all predicted values corresponding to the current time point
            values_at_t = flat_Cout[time == t]
            mean_values.append(np.mean(values_at_t))
            std_devs.append(np.std(values_at_t))
        
        # Convert to numpy arrays for plotting
        mean_values = np.array(mean_values)
        std_devs = np.array(std_devs)
        
        return unique_times, mean_values, std_devs
        
    def plot(self, figure=None, showPlot=False):
        if figure is None: figure = plt.figure()

        time, mean_values, std_devs = self.getMeanAndStd()
        
        # Plot
        ax = figure.add_subplot(111)
        
        ax.errorbar(time, mean_values, yerr=std_devs, fmt='-o', label='Mean ± Std Dev')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cpred')
        ax.set_title('Cpred Over Time')
        ax.legend()
        
        if showPlot:
            plt.show()
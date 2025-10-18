import time
import sys
import pickle
import torch.nn as nn
import numpy as np
import traceback
import os

from Container import InputHist, OutputHist
from NNAMPC import NN_A_MPC
from Common import parseInputs,tryToRemember, dataForOptipus, remember, getReactorModel

import Communication

PRINT_MSG = ""
def myPrintAndLog(msg):
    global PRINT_MSG
    PRINT_MSG += msg + "\n"
    print(msg)


TIME_OFFSET = 0
PREV_C3REF = None
INPUT_HIST = InputHist()
OUTPUT_HIST = OutputHist()
INTEGRATING_OFFSET = 0
DSE_OPTIMUM = None
ITERATION = 1

OUTPUT_HIST_REACTORMODEL = OutputHist()
REACOTR_MODEL = None
SIM_START_TIME = -10
def routine(dataFromOptipus, currentTime):
    global PREV_C3REF, INPUT_HIST, OUTPUT_HIST, INTEGRATING_OFFSET, DSE_OPTIMUM, ITERATION, OUTPUT_HIST_REACTORMODEL, REACOTR_MODEL, SIM_START_TIME
    
    myPrint = lambda msg: myPrintAndLog(msg)

    # Routine called by Optipus
    myPrint("NN-A-MPC script")
    myPrint("\tRequires 'C3meas', 'C3ref', 'temperature', 'flowRate'")
    
    # Inputs and time handling
    realTime = time.time()
    C3meas, C3ref, temperature, flowRate = parseInputs(dataFromOptipus) # 0, 1, 150, 1 
    
    myPrint(f"\tTime: {currentTime} Current time: {realTime} Iteration: {ITERATION} - C3meas: {C3meas}, C3ref: {C3ref}, temperature: {temperature}, flowRate: {flowRate}")        
    myPrint(f"\tInput history length: {len(INPUT_HIST.time)}\tOutput history length: {len(OUTPUT_HIST.time)}")

    
    # Reactor model for sentinel    
    myPrint("\t--- Simulate reactor model ---")
    if not REACOTR_MODEL: REACOTR_MODEL = getReactorModel(getModelForOptimization=False)
    #time_, Cot_, _ = REACOTR_MODEL.simulate(INPUT_HIST.time, INPUT_HIST.Cin, INPUT_HIST.flowRate, INPUT_HIST.temperature)
    # startTime, endTime, time_vec, Cin, flowRate, temperature=20, 
    endTime = OUTPUT_HIST.time[-1]
    myPrint(f"\t\t>>Simulate Reactor Model form: {SIM_START_TIME} to {endTime} (delta T: {endTime - SIM_START_TIME})")
    if endTime - SIM_START_TIME > 1e-2:
        time_, Cot_, _ = REACOTR_MODEL.simulateStep(SIM_START_TIME, endTime, INPUT_HIST.time, INPUT_HIST.Cin, INPUT_HIST.flowRate, INPUT_HIST.temperature)
        OUTPUT_HIST_REACTORMODEL.append(time_, Cot_)
        SIM_START_TIME = endTime
    else:
        myPrint("\t\t\tSimulation skipped")
    
    # Controller
    myPrint("\t--- NN-A-MPC Controller Step ---")
    C1, C2, DSE_OPTIMUM, INTEGRATING_OFFSET = NN_A_MPC(myPrint, currentTime, C3ref, temperature, flowRate, PREV_C3REF, INPUT_HIST, OUTPUT_HIST, INTEGRATING_OFFSET, OUTPUT_HIST_REACTORMODEL, DSE_OPTIMUM, ITERATION)

    myPrint(f"\tController outputs: C1: {C1}, C2: {C2}")
    INPUT_HIST.append([currentTime], np.array([[C1], [C2], [0], [0]]), flowRate, temperature) 
    
    timeNeeded = time.time() - realTime
    myPrint(f"\tController step time needed: {timeNeeded:.2f}s")
    myPrint(f"--- NN-A-MPC Controller Step finished ---")
    
    
    # Remember
    PREV_C3REF = C3ref

    return C1, C2, timeNeeded, OUTPUT_HIST_REACTORMODEL

NN_MPC_SMAPLETIME = 45
NEXT_NNMPC_TIME = 0

NEXT_SAMPLE_TIME = 0
SAMPLE_TIME = 15

def main():
    global TIME_OFFSET, OUTPUT_HIST, NEXT_NNMPC_TIME, INPUT_HIST, PRINT_MSG, NEXT_SAMPLE_TIME, ITERATION


    myPrintAndLog("-- Connect to OPC-UA Server ---")
    Communication.connectToOPCUAServer1()
    Communication.connectToOPCUAServer2()
    Communication.connectToOPCUAServer3()
    

    myPrintAndLog("-- Start main routine ---")
    TIME_OFFSET = time.time()

    if len(OUTPUT_HIST.time) <= 0: 
        OUTPUT_HIST.append([-10], np.array([[0], [0], [0], [0]]))
        OUTPUT_HIST.append([-.1], np.array([[0], [0], [0], [0]])) 

        
    if len(INPUT_HIST.time) <= 0: 
        INPUT_HIST.append([-10], np.array([[0], [0], [0], [0]]), 1, 150)
        INPUT_HIST.append([-.1], np.array([[0], [0], [0], [0]]), 1, 150) 

    while True:
        realTime = time.time()
        currentTime = realTime - TIME_OFFSET if TIME_OFFSET is not None else realTime

        if currentTime > NEXT_SAMPLE_TIME:
            NEXT_SAMPLE_TIME = currentTime + SAMPLE_TIME
            myPrintAndLog("-- Get data ---")
            try:
                data = Communication.readAllRequiredDataFromUPCUAServers() 
                C3meas, C3ref, temperature, flowRate = parseInputs(data)
                myPrintAndLog(str(data))
            except Exception as e:
                myPrintAndLog("[ERR] FAILED Getting data - keep old measured data")
            
            myPrintAndLog("\t>> Add to output hist")
            OUTPUT_HIST.append(currentTime, np.array([[0], [0], [C3meas], [0]]))
            myPrintAndLog(f"\tTime: {currentTime} Current time: {realTime} Iteration: {ITERATION} - C3meas: {C3meas}, C3ref: {C3ref}, temperature: {temperature}, flowRate: {flowRate}")        
        
            folder = ".\\logs"
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "sampled.csv"), "ab+") as f: 
                if currentTime < SAMPLE_TIME: f.write(f"Rel time;\tC3meas;\tC3ref;\tTemperature;\tFlowRate\n".encode('utf-8'))
                f.write(f"{float(currentTime):.1f};\t{float(C3meas):.4f};\t{float(C3ref):.4f};\t{float(temperature):.1f};\t{float(flowRate):.3f}\n".encode('utf-8'))



        if currentTime > NEXT_NNMPC_TIME:
            NEXT_NNMPC_TIME = currentTime + NN_MPC_SMAPLETIME
            myPrintAndLog(f"\t>> Do next NN-A-MPC at: {NEXT_NNMPC_TIME}")

            simOutputHist = None

            #try:
            if True:
                C1, C2, timeNeeded, simOutputHist = routine(data, currentTime)

                myPrintAndLog(f"\t\tC1: {C1:.4f}, C2: {C2:.4f} --> Write to OPC-UA")
                resultStr = Communication.writeAllValues(C1, C2, flowRate, temperature)
                myPrintAndLog(resultStr)

            #except Exception as e:
            #    C1 = None
            #    C2 = None
            #    timeNeeded = 0
            #    print(f"Shit: {str(e)}")

            # Reset print msg
            # Save data

            myPrintAndLog("---- LOG Data ----")
            folder = ".\\logs\\Iteration_" + str(ITERATION)  #+ "_Time_" + str(int(currentTime))
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "inputHist.pkl"), "wb") as f: pickle.dump(INPUT_HIST, f)
            with open(os.path.join(folder, "outputHist.pkl"), "wb") as f: pickle.dump(OUTPUT_HIST, f)
            with open(os.path.join(folder, "log.txt"), "wb") as f: f.write(PRINT_MSG.encode('utf-8'))
            if not simOutputHist: 
                with open(os.path.join(folder, "simOutputHist.pkl"), "wb") as f: pickle.dump(simOutputHist, f)


            folder = ".\\logs"
            with open(os.path.join(folder, "hist.csv"), "ab+") as f: 
                if ITERATION <= 1: f.write(f"Iteration;\tRel time;\tC3meas;\tC3ref;\tTemperature;\tFlowRate;\tC1;\tC2;\tIntegral comp state;\tCtrl time\n".encode('utf-8'))
                f.write(f"{ITERATION};\t{currentTime:.1f};\t{C3meas:.4f};\t{C3ref:.4f};\t{temperature:.1f};\t{flowRate:.3f};\t{C1:.4f};\t{C2:.4f};\t{INTEGRATING_OFFSET};\t{timeNeeded:.2f}\n".encode('utf-8'))

            PRINT_MSG = ""
            ITERATION += 1

            print(folder)
            myPrintAndLog("---- END LOG Data ----")


if __name__ == "__main__":

    main()

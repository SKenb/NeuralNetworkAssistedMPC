import pickle
import matplotlib.pyplot as plt


def main():
    dataContainer = []

    # Compare Polisher approaches
    #for filename in ["dataDSE_NO-Polisher.pkl", "dataDSE_SLSQP-Polisher.pkl", "dataDSE_COBYLA-Polisher.pkl"]: #"dataDSE_SLSQP-Polisher.pkl",
    
    # Compare approaches against measurement uncertainties
    subFolder = "./CompDumps/"
    for filename in []: # ADD dump filenames here
        with open(subFolder + filename, "rb") as f:
            dataContainer.append(pickle.load(f))


    figure = plt.figure(figsize=(7, 4))
    ax = figure.add_subplot(111)

    refPlotted = False
    smallestMaxTime = 1e6
    for data in dataContainer:
        if "C3ref" in data and not refPlotted:
            timeVec, C3ref = data["C3ref"]
            ax.plot(timeVec, C3ref, 'k--', label="Reference C3", linewidth=2)
            refPlotted = True

        #data['outputHist'].plot(showPlot=True)
        ax.plot(data['outputHist'].time, data['outputHist'].Cout[2, :], label=f"{data['name']}", linewidth=2)
        smallestMaxTime = min(smallestMaxTime, data['outputHist'].time[-1])


    ax.legend()
    #ax.grid()
    plt.xlim([0, smallestMaxTime])
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Comparison of different optimization approaches")
    
    plt.show()

if __name__ == "__main__":
    main()
    print("End")
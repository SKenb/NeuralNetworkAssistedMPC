import numpy as np
import matplotlib.pyplot as plt
import math

from DesignSpaceExploration import findMinimumCost

def plotDesignSpaceExploration(result, addOptimumMarker=True, useFlowRateVals=True):
    C1_vals = result.get("C1_vals")
    C2_vals = result.get("C2_vals")
    data_vals = result.get("flowRate_vals") if useFlowRateVals else result.get("temp_vals")
    costs = result.get("costs")
    numberOfSamples = result.get("numberOfCinSamples")

    # Plot
    unique_T = sorted(set(data_vals))
    n_heatmaps = len(unique_T) + 1 # for Heatmap

    n_cols = math.ceil(np.sqrt(n_heatmaps))
    n_rows = math.ceil(n_heatmaps / n_cols)

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows)) 

    # Heatmaps
    for idx, T in enumerate(unique_T):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Daten filtern
        x, y, z = [], [], []
        for c1, c2, t, cost in zip(C1_vals, C2_vals, data_vals, costs):
            if np.isclose(t, T):
                x.append(c1)
                y.append(c2)
                z.append(cost)

        # Gitter erstellen
        grid_x = np.linspace(min(x), max(x), numberOfSamples)
        grid_y = np.linspace(min(y), max(y), numberOfSamples)
        X, Y = np.meshgrid(grid_x, grid_y)
        Z = np.full_like(X, np.nan)

        for xi, yi, zi in zip(x, y, z):
            i_x = np.where(np.isclose(grid_x, xi))[0][0]
            i_y = np.where(np.isclose(grid_y, yi))[0][0]
            Z[i_y, i_x] = zi

        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', aspect='auto')
        ax.set_title(f"FR: = {T:.2f}mL/min" if useFlowRateVals else f"T = {T:.0f}°C")
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Cost")

    # 3D-Plot ganz unten, über alle Spalten
    ax3d = fig.add_subplot(n_rows, n_cols, n_rows * n_cols, projection='3d')
    sc = ax3d.scatter(C1_vals, C2_vals, data_vals, c=costs, cmap='viridis')
    ax3d.set_xlabel("C1")
    ax3d.set_ylabel("C2")
    ax3d.set_zlabel("Flow rate" if useFlowRateVals else "Temperature")

    if addOptimumMarker:    
        opt = findMinimumCost(result) 
        data_opt = opt.get("flowRate") if useFlowRateVals else opt.get("temp")

        ax3d.scatter(opt.get("C1"), opt.get("C2"), data_opt, c='red', s=80, marker='x', label='Optimum')
        ax3d.legend()

    fig.colorbar(sc, ax=ax3d, shrink=0.6, label='Cost')

    plt.tight_layout()
    plt.show()

def plotResultsOverDesignSpaceExplorationIterationsVeryCOnly(results, mode):
    n_plots = len(results) + 1
    n_cols = math.ceil(np.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
        
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for idx, result in enumerate(results):
        C1_vals = result.get("C1_vals")
        C2_vals = result.get("C2_vals")
        costs = result.get("costs")

        opt = findMinimumCost(result)

        if "flowRate_vals" in result:
            flowRate_vals = result.get("flowRate_vals")

            ax3d = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
            sc = ax3d.scatter(C1_vals, C2_vals, flowRate_vals, c=costs, cmap='viridis')
            ax3d.set_xlabel("C1")
            ax3d.set_ylabel("C2")
            ax3d.set_zlabel("Flow rate")

            ax3d.set_xlim(0, 1)
            ax3d.set_ylim(0, 1)
            ax3d.set_zlim(0.2, 2)

            ax3d.scatter(opt.get("C1"), opt.get("C2"), opt.get("flowRate"), c='red', s=80, marker='x', label='Optimum')
            ax3d.legend()

        elif "temp_vals" in result:
            temp_vals = result.get("temp_vals")

            ax3d = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
            sc = ax3d.scatter(C1_vals, C2_vals, temp_vals, c=costs, cmap='viridis')
            ax3d.set_xlabel("C1")
            ax3d.set_ylabel("C2")
            ax3d.set_zlabel("Temperature")

            ax3d.set_xlim(0, 1)
            ax3d.set_ylim(0, 1)
            ax3d.set_zlim(20, 300)

            ax3d.scatter(opt.get("C1"), opt.get("C2"), opt.get("temp"), c='red', s=80, marker='x', label='Optimum')
            ax3d.legend()

        else: 
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            tcf = ax.tricontourf(C1_vals, C2_vals, costs, levels=20, cmap="viridis")
            ax.scatter(opt.get("C1"), opt.get("C2"), color='red')
            ax.set_title(f"Result {idx}")
            ax.set_xlabel("C1")
            ax.set_ylabel("C2")
            ax.set_ylim((0, 1))
            ax.set_xlim((0, 1))
            plt.colorbar(tcf, ax=ax)

def plotResultsOverDesignSpaceExplorationIterations(results, useFlowRateVals=True):
    if results is None or len(results) <= 0: return

    n_plots = len(results) + 1 # + Trace of opt
    n_cols = math.ceil(np.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
        
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows)) 

    opt_C1_hist = []
    opt_C2_hist = []
    opt_flowRate_hist = []

    for idx, result in enumerate(results):
        C1_vals = result.get("C1_vals")
        C2_vals = result.get("C2_vals")
        data_vals = result.get("flowRate_vals") if useFlowRateVals else result.get("temp_vals")
        costs = result.get("costs")

        ax3d = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
        sc = ax3d.scatter(C1_vals, C2_vals, data_vals, c=costs, cmap='viridis')
        ax3d.set_xlabel("C1")
        ax3d.set_ylabel("C2")
        ax3d.set_zlabel("Flow rate" if useFlowRateVals else "Temperature")

        ax3d.set_xlim(0, 1)
        ax3d.set_ylim(0, 1)
        ax3d.set_zlim(0.2, 2)

        opt = findMinimumCost(result) 
        data_opt = opt.get("flowRate") if useFlowRateVals else opt.get("temp")

        opt_C1_hist.append(opt.get("C1"))
        opt_C2_hist.append(opt.get("C2"))
        opt_flowRate_hist.append(opt.get("flowRate"))

        ax3d.scatter(opt.get("C1"), opt.get("C2"), data_opt, c='red', s=80, marker='x', label='Optimum')
        ax3d.legend()

    ax3d = fig.add_subplot(n_rows, n_cols, n_rows*n_cols, projection='3d')
    ax3d.scatter(opt_C1_hist, opt_C2_hist, opt_flowRate_hist, c='red', s=80, marker='x', label='Optimum')
    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0.2, 2)

    fig.colorbar(sc, ax=ax3d, shrink=0.6, label='Cost')
    fig.show()
